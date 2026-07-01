# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""PPR / Naga-Zhang convergence tracking and patch-condition diagnostics.

PPRConvergenceTracker  — records A–F gradient errors across adaptive runs.
_save_patch_cond_hist  — saves a histogram of PPR patch condition numbers.
"""

from pathlib import Path

import numpy as np


class PPRConvergenceTracker:
    """Track PPR gradient-recovery convergence across multiple adaptive runs.

    After each completed adaptive run (at a given tolerance), call
    :meth:`record` to store six L² gradient errors:

      A = ||∇u_h − ∇u||              true FE gradient error
      B = ||∇u_h − G_h(u_h)||        PPR–FE residual
      C = ||G_h(u_h) − ∇u||          PPR recovery error (vs. exact)
      D = ||∇u − G_h(I²u)||          recovery error of the P2 interpolant
      E = ||G_h(u_h) − G_h(I²u)||    PPR sensitivity (PDE vs interpolation error)
      F = ||∇(I²u) − ∇u_h||          supercloseness

    The convergence variable is the global mesh vertex count ``n_verts``
    (N_v).  Each record also stores ``lambda3_max`` / ``lambda3_min`` — the
    extremes over all vertices of the smallest per-vertex Jacobian singular
    value λ₃,ₖ, i.e. the finest (layer-normal) mesh size.

    Expected rates (P2) — on a quasi-uniform 3-D mesh N_v ~ h⁻³, so an
    O(hᵖ) error decays like O(N_v^{-p/3}):
      A ~ O(h²),  B ~ O(h²),  C ~ O(h³) if superconvergent.
      D ~ O(h³),  E ~ o(h²),  F ~ O(h³)  (supercloseness).

    Parameters
    ----------
    layer_dir, layer_centre, layer_half_width :
        Retained for API compatibility (used by ``_save_patch_cond_hist``).
        No longer used by :meth:`record`, which now keys convergence on the
        vertex count rather than a boundary-layer h_z.
    """

    def __init__(
        self,
        layer_dir: int = 0,
        layer_centre: float = 0.0,
        layer_half_width: float = 0.05,
    ) -> None:
        self.layer_dir = layer_dir
        self.layer_centre = layer_centre
        self.layer_half_width = layer_half_width
        self._records: list[dict] = []

    # ------------------------------------------------------------------
    def record(
        self,
        tol: float,
        msh,
        u_h,
        u_numpy,
        Gh_func,
        grad_u_exact_factory,
        h_p: "np.ndarray | None" = None,
        interior_box: "tuple[float, float] | None" = None,
        interior_cylinder: "tuple[float, float, float, float] | None" = None,
    ) -> None:
        """Record one run's errors, vertex count and layer-normal mesh size.

        Parameters
        ----------
        tol :
            Equidistribution tolerance for this run.
        msh :
            Converged DOLFINx mesh.
        u_h :
            Converged FEM solution (P2).
        u_numpy :
            Exact solution as numpy callback ``u_numpy(x) -> array`` (scalar).
            Used to build the P2 interpolant I²u.
        Gh_func :
            Callable ``Gh_func(u_h) -> fem.Function`` returning the PPR
            recovered gradient (e.g. ``nz_eta_estimatorP2.Gh``).
        grad_u_exact_factory :
            Callable ``grad_u_exact_factory(msh) -> UFL vector expression``
            for the exact gradient ∇u.
        h_p :
            Deprecated / unused.  Retained so existing callers that pass the
            ``(n_vertices, tdim)`` array from :func:`adapt_h` keep working.
            The layer-normal mesh size is now taken from the per-vertex
            Jacobian singular values (λ₃,ₖ) instead.
        interior_box :
            Optional ``(lo, hi)`` scalar pair.  When provided, a second set
            of A–F errors is computed restricted to cells whose centroid
            satisfies ``lo <= x[d] <= hi`` for all dimensions d.  Use this
            to separate boundary-patch effects from bulk convergence.
            E.g. ``(-0.8, 0.8)`` on the [-1,1]^3 domain excludes the outer
            shell.  Set to ``None`` (default) to disable.
        interior_cylinder :
            Optional ``(R_lo, R_hi, z_lo, z_hi)`` tuple for domains with
            cylindrical geometry (e.g. tokamak).  Restricts A–F integrals to
            cells whose centroid satisfies ``R_lo <= R <= R_hi`` and
            ``z_lo <= z <= z_hi``, where ``R = sqrt(x²+y²)``.  Mutually
            exclusive with ``interior_box``; if both are set, the cylinder
            takes precedence.
        """
        from dolfinx import fem as _fem
        import ufl as _ufl
        from dolfinx.fem import form as _form
        from mpi4py import MPI as _MPI
        from eta_estimator1 import compute_lambda_P as _compute_lambda_P

        tdim = msh.topology.dim

        # ---- convergence variable: total number of mesh vertices -----
        # Replaces the old "median h_z in the layer" abscissa.  Uses the
        # global count so it is meaningful under MPI.
        n_verts = int(msh.topology.index_map(0).size_global)

        # ---- boundary-layer mesh size via the smallest singular value
        # lambda_3,k of the per-vertex Jacobian (k = vertex/patch index).
        # numpy SVD returns singular values in descending order, so column
        # index 2 is the finest direction — the layer normal.  We record its
        # extremes over all vertices: max_k lambda_3,k and min_k lambda_3,k.
        lambda_p = _compute_lambda_P(u_h)          # (n_vertices, tdim)
        lam3 = lambda_p[:, 2]
        if lam3.size:
            lam3_max_loc, lam3_min_loc = float(lam3.max()), float(lam3.min())
        else:
            lam3_max_loc, lam3_min_loc = -np.inf, np.inf
        lambda3_max = msh.comm.allreduce(lam3_max_loc, op=_MPI.MAX)
        lambda3_min = msh.comm.allreduce(lam3_min_loc, op=_MPI.MIN)

        # ---- Build I²u: P2 interpolant of the exact solution ---------
        V = u_h.function_space
        I2u = _fem.Function(V)
        I2u.interpolate(u_numpy)

        # ---- Exact gradient as UFL expression ------------------------
        grad_u_ex = grad_u_exact_factory(msh)

        # ---- PPR recovered gradients ---------------------------------
        G_nz   = Gh_func(u_h)   # G_h(u_h)  — recovery of FE solution
        Gh_I2u = Gh_func(I2u)   # G_h(I²u)  — recovery of interpolant

        def _l2(expr):
            val = _fem.assemble_scalar(_form(_ufl.inner(expr, expr) * _ufl.dx))
            return float(np.sqrt(val))

        # ---- A, B, C (original three) --------------------------------
        A = _l2(_ufl.grad(u_h) - grad_u_ex)          # ||∇u_h − ∇u||
        B = _l2(_ufl.grad(u_h) - G_nz)               # ||∇u_h − G_h(u_h)||
        C = _l2(G_nz - grad_u_ex)                    # ||G_h(u_h) − ∇u||

        # ---- D, E, F (interpolant decomposition) --------------------
        D = _l2(grad_u_ex - Gh_I2u)                  # ||∇u − G_h(I²u)||
        E = _l2(G_nz - Gh_I2u)                       # ||G_h(u_h) − G_h(I²u)||
        F = _l2(_ufl.grad(I2u) - _ufl.grad(u_h))    # ||∇(I²u) − ∇u_h||

        entry = {"tol": tol, "n_verts": n_verts,
                 "lambda3_max": lambda3_max, "lambda3_min": lambda3_min,
                 "A": A, "B": B, "C": C, "D": D, "E": E, "F": F}
        self._records.append(entry)
        print(
            f"  [PPRTracker] tol={tol:.3e}  N_v={n_verts}"
            f"  λ3∈[{lambda3_min:.3e}, {lambda3_max:.3e}]"
        )
        print(
            f"  [PPRTracker]  A={A:.3e}  B={B:.3e}  C={C:.3e}"
        )
        print(
            f"  [PPRTracker]  D={D:.3e}  E={E:.3e}  F={F:.3e}"
        )

        # ── interior-box errors ──────────────────────────────────────
        if interior_box is not None:
            lo, hi = interior_box
            msh.topology.create_connectivity(tdim, 0)
            _ctv  = msh.topology.connectivity(tdim, 0).array.reshape(-1, tdim + 1)
            _bary = coords[_ctv].mean(axis=1)          # (n_cells, gdim)
            _mask = np.all((_bary >= lo) & (_bary <= hi), axis=1)
            _int_cells = np.where(_mask)[0].astype(np.int32)
            import dolfinx.mesh as _dmesh
            _ct = _dmesh.meshtags(msh, tdim, _int_cells,
                                  np.ones(len(_int_cells), dtype=np.int32))
            _dx_int = _ufl.Measure("dx", domain=msh, subdomain_data=_ct)(1)

            def _l2_int(expr):
                val = _fem.assemble_scalar(_form(_ufl.inner(expr, expr) * _dx_int))
                return float(np.sqrt(val))

            Ai = _l2_int(_ufl.grad(u_h) - grad_u_ex)
            Bi = _l2_int(_ufl.grad(u_h) - G_nz)
            Ci = _l2_int(G_nz - grad_u_ex)
            Di = _l2_int(grad_u_ex - Gh_I2u)
            Ei = _l2_int(G_nz - Gh_I2u)
            Fi = _l2_int(_ufl.grad(I2u) - _ufl.grad(u_h))

            entry.update({"A_int": Ai, "B_int": Bi, "C_int": Ci,
                          "D_int": Di, "E_int": Ei, "F_int": Fi})
            n_int = int(len(_int_cells))
            n_tot = int(msh.topology.index_map(tdim).size_local)
            print(
                f"  [PPRTracker/int] box=[{lo},{hi}]^3  "
                f"{n_int}/{n_tot} cells"
            )
            print(
                f"  [PPRTracker/int]  A={Ai:.3e}  B={Bi:.3e}  C={Ci:.3e}"
            )
            print(
                f"  [PPRTracker/int]  D={Di:.3e}  E={Ei:.3e}  F={Fi:.3e}"
            )
        # ── interior-cylinder errors (cylindrical-geometry domains) ──
        if interior_cylinder is not None:
            R_lo, R_hi, z_lo, z_hi = interior_cylinder
            msh.topology.create_connectivity(tdim, 0)
            _ctv  = msh.topology.connectivity(tdim, 0).array.reshape(-1, tdim + 1)
            _bary = coords[_ctv].mean(axis=1)          # (n_cells, 3)
            _R    = np.sqrt(_bary[:, 0]**2 + _bary[:, 1]**2)
            _mask = (_R >= R_lo) & (_R <= R_hi) & (_bary[:, 2] >= z_lo) & (_bary[:, 2] <= z_hi)
            _int_cells = np.where(_mask)[0].astype(np.int32)
            import dolfinx.mesh as _dmesh
            _ct = _dmesh.meshtags(msh, tdim, _int_cells,
                                  np.ones(len(_int_cells), dtype=np.int32))
            _dx_int = _ufl.Measure("dx", domain=msh, subdomain_data=_ct)(1)

            def _l2_cyl(expr):
                val = _fem.assemble_scalar(_form(_ufl.inner(expr, expr) * _dx_int))
                return float(np.sqrt(val))

            Ai = _l2_cyl(_ufl.grad(u_h) - grad_u_ex)
            Bi = _l2_cyl(_ufl.grad(u_h) - G_nz)
            Ci = _l2_cyl(G_nz - grad_u_ex)
            Di = _l2_cyl(grad_u_ex - Gh_I2u)
            Ei = _l2_cyl(G_nz - Gh_I2u)
            Fi = _l2_cyl(_ufl.grad(I2u) - _ufl.grad(u_h))

            entry.update({"A_int": Ai, "B_int": Bi, "C_int": Ci,
                          "D_int": Di, "E_int": Ei, "F_int": Fi})
            n_int = int(len(_int_cells))
            n_tot = int(msh.topology.index_map(tdim).size_local)
            print(
                f"  [PPRTracker/cyl] R=[{R_lo},{R_hi}]  z=[{z_lo},{z_hi}]  "
                f"{n_int}/{n_tot} cells"
            )
            print(f"  [PPRTracker/cyl]  A={Ai:.3e}  B={Bi:.3e}  C={Ci:.3e}")
            print(f"  [PPRTracker/cyl]  D={Di:.3e}  E={Ei:.3e}  F={Fi:.3e}")

    # ------------------------------------------------------------------
    def _sorted(self) -> list[dict]:
        return sorted(self._records, key=lambda r: r["n_verts"])

    def print_table(self) -> None:
        """Print a convergence table sorted by ascending N_v (vertex count).

        Convergence rates are computed with respect to log(N_v): for an
        O(h^p) error on a quasi-uniform 3-D mesh (N_v ~ h^{-3}) the reported
        rate is ~ -p/3.
        """
        recs = self._sorted()

        def _print_block(label, keys):
            qs = list(keys)
            header = (
                f"{'tol':>10}  {'N_v':>10}  "
                + "  ".join(f"{q:>10}" for q in qs)
                + "  "
                + "  ".join(f"{'r'+q:>6}" for q in qs)
            )
            sep = "─" * len(header)
            print(f"\n{label}")
            print(header)
            print(sep)
            for k, r in enumerate(recs):
                if k == 0 or any(r.get(q) is None for q in qs):
                    rate_str = "  ".join(f"{'—':>6}" for _ in qs)
                else:
                    prev = recs[k - 1]
                    log_n = np.log(r["n_verts"] / prev["n_verts"])
                    rates = [
                        np.log(r[q] / prev[q]) / log_n if log_n != 0 else float("nan")
                        for q in qs
                    ]
                    rate_str = "  ".join(f"{s:6.2f}" for s in rates)
                vals_str = "  ".join(f"{r.get(q, float('nan')):>10.3e}" for q in qs)
                print(
                    f"  {r['tol']:>8.3e}  {r['n_verts']:>10d}"
                    f"  {vals_str}  {rate_str}"
                )

        _print_block("Full domain", ("A", "B", "C", "D", "E", "F"))

        if recs and "A_int" in recs[0]:
            _print_block("Interior box only",
                         ("A_int", "B_int", "C_int", "D_int", "E_int", "F_int"))

    def to_csv(self, filename: str = "ppr_convergence.csv") -> None:
        """Dump all recorded runs to a CSV, sorted by ascending N_v.

        Columns: tol, n_verts, lambda3_max, lambda3_min, A, B, C, D, E, F,
        and (if present) A_int..F_int. Missing optional columns are left
        blank. Use ``NZ_diagnosis/plot_ppr_convergence.py`` to turn this CSV
        into figures.

        ``n_verts`` is the global mesh vertex count (the convergence
        variable); ``lambda3_max``/``lambda3_min`` are the extremes over
        all vertices of the smallest per-vertex Jacobian singular value
        (the boundary-layer / finest-direction mesh size).
        """
        import csv

        recs = self._sorted()
        if len(recs) == 0:
            print("  [PPRTracker] No records yet — nothing to write.")
            return

        fields = ["tol", "n_verts", "lambda3_max", "lambda3_min",
                  "A", "B", "C", "D", "E", "F"]
        if any("A_int" in r for r in recs):
            fields += ["A_int", "B_int", "C_int", "D_int", "E_int", "F_int"]

        out = Path(filename).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for r in recs:
                writer.writerow({k: r.get(k, "") for k in fields})
        print(f"  [PPRTracker] Saved → {out}")

def _save_patch_cond_hist(u_h, msh, loop_idx, out_dir, cond_params):
    """Compute PPR patch condition numbers and save a stacked histogram.

    Vertices are split into three categories coloured in each bar:
      - red:   boundary (∂Ω)
      - blue:  layer interior (|x[layer_dir] - layer_centre| < layer_half_width)
      - green: other interior

    Parameters
    ----------
    cond_params : dict with keys layer_dir, layer_centre, layer_half_width
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nz_eta_estimatorP2 import Gh_with_cond
    from nz_mesh_helpers import find_boundary

    _, cond_arr = Gh_with_cond(u_h)

    coords = msh.geometry.x
    n_verts = len(cond_arr)
    B = find_boundary(msh)                      # bool[n_vertices], True = on ∂Ω

    ld = cond_params["layer_dir"]
    lc = cond_params["layer_centre"]
    lw = cond_params["layer_half_width"]
    in_layer = np.abs(coords[:n_verts, ld] - lc) < lw

    # 0 = boundary (takes priority), 1 = layer interior, 2 = other interior
    cats = np.full(n_verts, 2, dtype=np.int32)
    cats[in_layer & ~B] = 1
    cats[B] = 0

    log_cond = np.log10(np.maximum(cond_arr, 1.0))
    cond_max = float(log_cond.max())
    bins = np.linspace(0.0, max(cond_max + 0.5, 8.0), 60)

    n_bnd   = int(np.sum(cats == 0))
    n_layer = int(np.sum(cats == 1))
    n_other = int(np.sum(cats == 2))
    med  = float(np.median(log_cond))
    p95  = float(np.percentile(log_cond, 95))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        [log_cond[cats == 0], log_cond[cats == 1], log_cond[cats == 2]],
        bins=bins, stacked=True,
        color=["#d62728", "#1f77b4", "#2ca02c"],
        label=[
            f"boundary ({n_bnd} vertices)",
            f"layer |x[{ld}]−{lc:.2g}|<{lw:.2g} ({n_layer} vertices)",
            f"other interior ({n_other} vertices)",
        ],
        edgecolor="none", alpha=0.85,
    )
    ax.axvline(med, color="black", ls="--", lw=1.5,
               label=f"median κ = {10**med:.2e}")
    ax.axvline(p95, color="black", ls=":",  lw=1.5,
               label=f"95th pct κ = {10**p95:.2e}")
    ax.set_xlabel(r"$\log_{10}\,\kappa(A^T A)$", fontsize=13)
    ax.set_ylabel(r"$N_v$", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    hist_dir = Path(out_dir) / "patch_cond_hists"
    hist_dir.mkdir(exist_ok=True)
    p = hist_dir / f"patch_cond_hist_loop{loop_idx:03d}.png"
    fig.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(
        f"  [cond_hist] loop {loop_idx + 1}  "
        f"median κ={10**med:.2e}  p95 κ={10**p95:.2e}  → patch_cond_hists/{p.name}"
    )
