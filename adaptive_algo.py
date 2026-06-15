# pyright: reportMissingImports=false, reportMissingModuleSource=false
import csv
import os
import shutil
from pathlib import Path
from typing import Callable

import numpy as np

from solver import solve_poisson as _default_solver, u_exact_np as _default_u_exact
from mesh_helpers import (
    write_cube_mesh,
    generate_initial_mesh,
    read_medit_to_dolfinx,
    to_vtu,
    build_dolfinx_to_medit_map,
    build_metric,
    adapt_mesh_mmg,
    save_computed_quantities,
    snap_tokamak_wall_vertices,
    get_boundary_vertex_indices,
)
from eta_estimator1 import (
    compute_jacobian_svd,
    adapt_h,
    compute_anisotropic_eta,
    compute_G_tilde,
    compute_G_tilde_nz,
    compute_G_P,
    compute_lambda_P,
    compute_sigma_P,
    compute_gradient_dg0,
)
from error_metrics import compute_error_metrics, compute_error_metrics_ZZ, compute_error_norms


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

    Expected rates (P2, w.r.t. h_z in the boundary layer):
      A ~ O(h²),  B ~ O(h²),  C ~ O(h³) if superconvergent.
      D ~ O(h³),  E ~ o(h²),  F ~ O(h³)  (supercloseness).

    Parameters
    ----------
    layer_dir :
        Coordinate index (0=x, 1=y, 2=z) that defines the layer normal.
    layer_centre :
        Location of the layer centre along ``layer_dir``.
    layer_half_width :
        Half-width of the layer used to select representative vertices.
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
    ) -> None:
        """Record one run's errors and representative h_z.

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
            Optional ``(n_vertices, tdim)`` array from :func:`adapt_h`.
            When provided ``h_p[:, 2]`` is used for h_z; otherwise h_z
            is estimated from each in-layer cell's z-vertex range.
        interior_box :
            Optional ``(lo, hi)`` scalar pair.  When provided, a second set
            of A–F errors is computed restricted to cells whose centroid
            satisfies ``lo <= x[d] <= hi`` for all dimensions d.  Use this
            to separate boundary-patch effects from bulk convergence.
            E.g. ``(-0.8, 0.8)`` on the [-1,1]^3 domain excludes the outer
            shell.  Set to ``None`` (default) to disable.
        """
        from dolfinx import fem as _fem
        import ufl as _ufl
        from dolfinx.fem import form as _form

        tdim = msh.topology.dim
        coords = msh.geometry.x

        # ---- representative h_z in the boundary layer ----------------
        vert_in_layer = (
            np.abs(coords[:, self.layer_dir] - self.layer_centre)
            < self.layer_half_width
        )
        if h_p is not None:
            h_z_vals = h_p[vert_in_layer, 2]
        else:
            msh.topology.create_connectivity(tdim, 0)
            ctv = msh.topology.connectivity(tdim, 0).array.reshape(
                -1, tdim + 1
            )
            bary_dir = coords[ctv, self.layer_dir].mean(axis=1)
            cell_in_layer = (
                np.abs(bary_dir - self.layer_centre) < self.layer_half_width
            )
            z_c = coords[:, 2]
            h_z_cells = z_c[ctv].max(axis=1) - z_c[ctv].min(axis=1)
            h_z_vals = h_z_cells[cell_in_layer]

        if len(h_z_vals) == 0:
            print(
                f"  [PPRTracker] WARNING: no mesh entities found in layer "
                f"for tol={tol:.3e}; skipping record."
            )
            return
        h_layer = float(np.median(h_z_vals))

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

        entry = {"tol": tol, "h_layer": h_layer,
                 "A": A, "B": B, "C": C, "D": D, "E": E, "F": F}
        self._records.append(entry)
        print(
            f"  [PPRTracker] tol={tol:.3e}  h_z={h_layer:.3e}"
            f"  A={A:.3e}  B={B:.3e}  C={C:.3e}"
        )
        print(
            f"  [PPRTracker]  D={D:.3e}  E={E:.3e}  F={F:.3e}"
        )

        # ── interior-box errors (delete this block to restore original) ──
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
        # ─────────────────────────────────────────────────────────────────

    # ------------------------------------------------------------------
    def _sorted(self) -> list[dict]:
        return sorted(self._records, key=lambda r: r["h_layer"])

    def print_table(self) -> None:
        """Print a convergence table sorted by ascending h_z."""
        recs = self._sorted()

        def _print_block(label, keys):
            qs = list(keys)
            header = (
                f"{'tol':>10}  {'h_z':>10}  "
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
                    log_h = np.log(r["h_layer"] / prev["h_layer"])
                    rates = [
                        np.log(r[q] / prev[q]) / log_h
                        for q in qs
                    ]
                    rate_str = "  ".join(f"{s:6.2f}" for s in rates)
                vals_str = "  ".join(f"{r.get(q, float('nan')):>10.3e}" for q in qs)
                print(
                    f"  {r['tol']:>8.3e}  {r['h_layer']:>10.3e}"
                    f"  {vals_str}  {rate_str}"
                )

        _print_block("Full domain", ("A", "B", "C", "D", "E", "F"))

        # ── interior-box table (delete this block to restore original) ───
        if recs and "A_int" in recs[0]:
            _print_block("Interior box only",
                         ("A_int", "B_int", "C_int", "D_int", "E_int", "F_int"))
        # ─────────────────────────────────────────────────────────────────

    def plot(self, filename: str = "ppr_convergence.pdf") -> None:
        """Save three log–log convergence plots derived from *filename*.

        Generates ``<stem>_ABC``, ``<stem>_DEF``, and ``<stem>_all``
        as both PDF and PNG.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        recs = self._sorted()
        if len(recs) == 0:
            print("  [PPRTracker] No records yet — nothing to plot.")
            return

        h = np.array([r["h_layer"] for r in recs])

        _fields = {
            "A": {"label": r"$A=\|\nabla u_h-\nabla u\|$",
                  "color": "#1f77b4", "marker": "o", "ls": "-"},
            "B": {"label": r"$B=\|\nabla u_h-G_h u_h\|$",
                  "color": "#ff7f0e", "marker": "s", "ls": "-"},
            "C": {"label": r"$C=\|G_h u_h-\nabla u\|$",
                  "color": "#2ca02c", "marker": "^", "ls": "-"},
            "D": {"label": r"$D=\|\nabla u-G_h(I^2 u)\|$",
                  "color": "#d62728", "marker": "D", "ls": "--"},
            "E": {"label": r"$E=\|G_h u_h-G_h(I^2 u)\|$",
                  "color": "#9467bd", "marker": "v", "ls": "--"},
            "F": {"label": r"$F=\|\nabla(I^2 u)-\nabla u_h\|$",
                  "color": "#8c564b", "marker": "P", "ls": "--"},
        }
        _yoff = {"A": 1.4, "B": 0.6, "C": 1.4, "D": 0.6, "E": 1.4, "F": 0.6}

        def _make_fig(subset, title):
            fig, ax = plt.subplots(figsize=(8, 6))
            for name in subset:
                info = _fields[name]
                vals = np.array([r[name] for r in recs])
                if np.all(vals > 0):
                    ax.loglog(h, vals,
                              marker=info["marker"], linestyle=info["ls"],
                              color=info["color"], linewidth=2, markersize=7,
                              label=info["label"])
            if len(recs) >= 2:
                h_ref = np.array([h.min(), h.max()])
                A_fine = float(recs[-1]["A"])
                ax.loglog(h_ref, (A_fine / h[-1]**2) * h_ref**2,
                          "--", color="gray", lw=1, alpha=0.5, label=r"$O(h^2)$")
                D_fine = float(recs[-1]["D"])
                if D_fine > 0:
                    ax.loglog(h_ref, (D_fine / h[-1]**3) * h_ref**3,
                              ":", color="gray", lw=1, alpha=0.5, label=r"$O(h^3)$")
                for name in subset:
                    vals = np.array([r[name] for r in recs])
                    if not np.all(vals > 0):
                        continue
                    rate = np.log(vals[-1] / vals[-2]) / np.log(h[-1] / h[-2])
                    xmid = np.sqrt(h[-1] * h[-2])
                    ymid = np.sqrt(vals[-1] * vals[-2])
                    ax.annotate(f"{rate:.2f}", xy=(xmid, ymid * _yoff[name]),
                                fontsize=9, color=_fields[name]["color"],
                                fontweight="bold", ha="center")
            ax.set_xlabel(r"$h_z$ (median in boundary layer)", fontsize=13)
            ax.set_ylabel(r"$L^2$ gradient error", fontsize=13)
            ax.set_title(title, fontsize=14)
            ncol = 1 if len(subset) <= 3 else 2
            ax.legend(fontsize=9, loc="upper left", ncol=ncol)
            ax.grid(True, which="both", alpha=0.3)
            fig.tight_layout()
            return fig

        out = Path(filename).resolve()
        parent = out.parent
        stem = out.stem

        for subset, title, suffix in [
            (["A", "B", "C"],
             "PPR convergence: FE errors (P2)",
             "_ABC"),
            (["D", "E", "F"],
             "PPR convergence: interpolant decomposition (P2)",
             "_DEF"),
            (list(_fields.keys()),
             "PPR gradient-recovery convergence (P2)",
             "_all"),
        ]:
            fig = _make_fig(subset, title)
            for ext in (".pdf", ".png"):
                p = parent / (stem + suffix + ext)
                fig.savefig(str(p), dpi=150, bbox_inches="tight")
                print(f"  [PPRTracker] Saved → {p}")
            plt.close(fig)

        # ── interior-box plots (delete this block to restore original) ───
        if recs and "A_int" in recs[0]:
            # Remap _fields keys to their _int counterparts for _make_fig
            _fields_int = {
                q + "_int": {**info, "label": info["label"].replace("$", "$") + " (int)"}
                for q, info in _fields.items()
            }
            _yoff_int = {q + "_int": v for q, v in _yoff.items()}

            def _make_fig_int(subset_int, title):
                fig, ax = plt.subplots(figsize=(8, 6))
                for key in subset_int:
                    base = key.replace("_int", "")
                    info = _fields[base]
                    vals = np.array([r.get(key, float("nan")) for r in recs])
                    if np.all(np.isfinite(vals)) and np.all(vals > 0):
                        ax.loglog(h, vals,
                                  marker=info["marker"], linestyle=info["ls"],
                                  color=info["color"], linewidth=2, markersize=7,
                                  label=info["label"] + " (int)")
                if len(recs) >= 2:
                    h_ref = np.array([h.min(), h.max()])
                    A_fine = float(recs[-1].get("A_int", recs[-1]["A"]))
                    ax.loglog(h_ref, (A_fine / h[-1]**2) * h_ref**2,
                              "--", color="gray", lw=1, alpha=0.5, label=r"$O(h^2)$")
                    D_fine = float(recs[-1].get("D_int", recs[-1]["D"]))
                    if D_fine > 0:
                        ax.loglog(h_ref, (D_fine / h[-1]**3) * h_ref**3,
                                  ":", color="gray", lw=1, alpha=0.5, label=r"$O(h^3)$")
                    for key in subset_int:
                        base = key.replace("_int", "")
                        vals = np.array([r.get(key, float("nan")) for r in recs])
                        if not (np.all(np.isfinite(vals)) and np.all(vals > 0)):
                            continue
                        rate = np.log(vals[-1] / vals[-2]) / np.log(h[-1] / h[-2])
                        xmid = np.sqrt(h[-1] * h[-2])
                        ymid = np.sqrt(vals[-1] * vals[-2])
                        ax.annotate(f"{rate:.2f}",
                                    xy=(xmid, ymid * _yoff[base]),
                                    fontsize=9, color=_fields[base]["color"],
                                    fontweight="bold", ha="center")
                ax.set_xlabel(r"$h_z$ (median in boundary layer)", fontsize=13)
                ax.set_ylabel(r"$L^2$ gradient error", fontsize=13)
                ax.set_title(title, fontsize=14)
                ncol = 1 if len(subset_int) <= 3 else 2
                ax.legend(fontsize=9, loc="upper left", ncol=ncol)
                ax.grid(True, which="both", alpha=0.3)
                fig.tight_layout()
                return fig

            for subset_int, title, suffix in [
                (["A_int", "B_int", "C_int"],
                 "PPR convergence: FE errors — interior only (P2)",
                 "_ABC_int"),
                (["D_int", "E_int", "F_int"],
                 "PPR convergence: interpolant decomp. — interior only (P2)",
                 "_DEF_int"),
                (["A_int", "B_int", "C_int", "D_int", "E_int", "F_int"],
                 "PPR convergence: all errors — interior only (P2)",
                 "_all_int"),
            ]:
                fig = _make_fig_int(subset_int, title)
                for ext in (".pdf", ".png"):
                    p = parent / (stem + suffix + ext)
                    fig.savefig(str(p), dpi=150, bbox_inches="tight")
                    print(f"  [PPRTracker] Saved → {p}")
                plt.close(fig)
        # ─────────────────────────────────────────────────────────────────


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
    ax.set_ylabel("Number of vertices", fontsize=13)
    ax.set_title(
        f"PPR patch condition numbers — loop {loop_idx + 1}  "
        f"(n={n_verts} vertices)",
        fontsize=13,
    )
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


def run_adaptive_poisson(
    results_dir: Path | str,
    solver: Callable | None = None,
    u_numpy: Callable[[np.ndarray], np.ndarray] | None = None,
    n_loop: int = 5,
    hmax: float = 1.0,
    hmin: float = 1e-10,
    hgrad: float = -1,
    tol: float = 0.5,
    alpha: float = 0.25,
    correction_factor: float = 1.5,
    degree_raise: int = 4,
    mmg3d_exe: str = "/usr/local/bin/mmg3d_O3",
    initial_mesh_file: Path | str | None = None,
    k: int = 1,
    mmg_extra_args: list | None = None,
    tok_snap: dict | None = None,
    patch_cond_diag: "dict | None" = None,
) -> tuple:
    """Run the anisotropic adaptive Poisson algorithm (Table 1.6).

    Parameters
    ----------
    results_dir :
        Root output directory.  Sub-directories will be created automatically:
        ``meshes/``, ``sols/``, ``vtk/``, ``compute_errs/``.
    solver :
        Callable ``solver(msh) -> (u_h, f_rhs)``.  Defaults to
        :func:`solver.solve_poisson` (tanh boundary-layer problem).
    u_numpy :
        Exact solution callback ``u_numpy(x)`` used for error metrics.
        Defaults to :func:`solver.u_exact_np`.
    n_loop :
        Number of adaptive iterations.
    hmax, hmin :
        Upper / lower bounds for mesh size passed to MMG3D.
    hgrad :
        Mesh-size gradation factor for MMG3D (``-1`` disables it).
    tol :
        Equidistribution tolerance.
    alpha :
        Half-width of the equidistribution acceptance band.
    correction_factor :
        Multiplier applied when coarsening or refining ``h_p``.
    degree_raise :
        Degree increment used in error-norm projection space.
    mmg3d_exe :
        Path to the MMG3D executable.
    initial_mesh_file :
        Path to an existing Medit ``.mesh`` file to use as the starting mesh
        instead of generating one from scratch.  When ``None`` (default) the
        initial mesh is generated by MMG3D from a coarse cube.
    tok_snap :
        When not ``None``, snap tokamak boundary vertices onto the exact
        cylindrical walls after each mesh generation.  Must be a dict with
        keys ``r_inner`` (float) and ``r_outer`` (float).

    Returns
    -------
    msh :
        Final mesh (DOLFINx).
    u_h :
        Final FEM solution.
    h_p :
        Final target mesh-size array, shape ``(n_vertices, tdim)``.
    final_metrics :
        Dictionary with final-iteration metrics and estimators.
    k :
        Polynomial degree and estimator selector.  ``k=1`` uses the ZZ
        recovered gradient; ``k=2`` uses the Naga-Zhang (PPR) estimator.
    iteration_metrics :
        List of per-iteration dicts with keys ``loop_idx``, ``n_vertices``,
        and ``TRE``.
    """
    if solver is None:
        solver = _default_solver
    if u_numpy is None:
        u_numpy = _default_u_exact

    results_dir = Path(results_dir)
    meshes_dir       = results_dir / "meshes"
    sols_dir         = results_dir / "sols"
    vtk_dir          = results_dir / "vtk"
    compute_errs_dir = results_dir / "compute_errs"

    for d in (results_dir, meshes_dir, sols_dir, vtk_dir, compute_errs_dir):
        d.mkdir(exist_ok=True, parents=True)

    mmg_log_file = results_dir / "mmg_output.txt"

    print(f"[INIT] Results directory: {results_dir}")
    print(f"       meshes → {meshes_dir}")
    print(f"       sols   → {sols_dir}")
    print(f"       vtk    → {vtk_dir}")
    print(f"       errs   → {compute_errs_dir}")

    # ------------------------------------------------------------------
    # STEP 1: Generate initial mesh or copy provided one
    # ------------------------------------------------------------------
    mesh_0_path = str(meshes_dir / "mesh_0.mesh")

    if initial_mesh_file is None:
        print("\n[STEP 1] Generating initial mesh...")
        coarse_mesh_path = str(meshes_dir / "mesh_coarse.mesh")
        write_cube_mesh(coarse_mesh_path)
        if not os.path.isfile(coarse_mesh_path):
            raise FileNotFoundError(f"Coarse mesh not created: {coarse_mesh_path}")
        generate_initial_mesh(coarse_mesh_path, mesh_0_path, mmg3d_exe, hmax=hmax, hgrad=-1)
        if not os.path.isfile(mesh_0_path):
            raise FileNotFoundError(f"Initial mesh not created by MMG: {mesh_0_path}")
    else:
        print(f"\n[STEP 1] Using provided initial mesh: {initial_mesh_file}")
        shutil.copy2(str(initial_mesh_file), mesh_0_path)

    # ---- Snap statistics CSVs (one per face, written/appended each snap) ----
    _SNAP_FACES = ("inner", "outer", "bottom", "top")
    _snap_csv: dict[str, Path] = {}
    if tok_snap is not None:
        _CSV_HDR = ["mesh_idx", "total", "snapped", "bisected",
                    "mean_dist_bisected", "std_dist_bisected", "max_dist_bisected"]
        for _face in _SNAP_FACES:
            _p = results_dir / f"snap_stats_{_face}.csv"
            _snap_csv[_face] = _p
            with open(_p, "w", newline="") as _f:
                csv.writer(_f).writerow(_CSV_HDR)

    def _write_snap_row(mesh_idx: int, snap_result: dict) -> None:
        for _face in _SNAP_FACES:
            s = snap_result[_face]
            with open(_snap_csv[_face], "a", newline="") as _f:
                csv.writer(_f).writerow([
                    mesh_idx, s["total"], s["snapped"], s["bisected"],
                    s["mean_dist"], s["std_dist"], s["max_dist"],
                ])

    if tok_snap is not None:
        _snap_result = snap_tokamak_wall_vertices(
            mesh_0_path,
            r_inner=tok_snap["r_inner"],
            r_outer=tok_snap["r_outer"],
        )
        _write_snap_row(0, _snap_result)

    msh = read_medit_to_dolfinx(mesh_0_path)
    print(
        f"  Initial mesh: "
        f"{msh.topology.index_map(3).size_global} cells, "
        f"{msh.topology.index_map(0).size_global} vertices"
    )

    vtu_path = to_vtu(mesh_0_path, output_dir=vtk_dir)
    if vtu_path:
        print(f"  VTU saved: {Path(vtu_path).name}")

    # ------------------------------------------------------------------
    # MAIN ADAPTIVE LOOP
    # ------------------------------------------------------------------
    u_h = None
    h_p = None
    final_metrics = {}
    iteration_metrics = []

    for loop_idx in range(n_loop):
        print(f"\n{'='*70}")
        print(f"LOOP {loop_idx + 1}/{n_loop}")
        print(f"{'='*70}")

        # ---- Step 2: Solve ----------------------------------------
        print("\n[2] Solving Poisson equation...")
        u_h, f_rhs = solver(msh)
        tdim = msh.topology.dim
        n_vertices_iter = int(msh.topology.index_map(0).size_global)
        n_cells_iter    = int(msh.topology.index_map(tdim).size_global)
        print(f"  Solution on {n_cells_iter} cells, {n_vertices_iter} vertices")

        # ---- Per-iteration TRE ----------------------------------------
        print("\n[2b] Computing TRE...")
        norm_grad_e, norm_grad_u, norm_grad_uh = compute_error_norms(
            u_h, u_numpy, degree_raise
        )
        tre_iter = float(norm_grad_e / norm_grad_uh) if norm_grad_uh > 1e-30 else float('nan')
        print(f"  TRE={tre_iter:.6e}  norm_grad_u={norm_grad_u:.6e}  norm_grad_uh={norm_grad_uh:.6e}  n_vertices={n_vertices_iter}")
        if norm_grad_u < 1e-30:
            print("  WARNING: exact solution norm is ~0 — problem is degenerate, stopping loop.")
            break
# ---- Step 2c: PPR diagnostic (remove after debugging) -----
        if k == 2:
            from nz_eta_estimatorP2 import Gh as _Gh_diag
            from diagnose_ppr import diagnose_ppr
            print(f"\n[2c] PPR diagnostic (loop {loop_idx}):")
            diagnose_ppr(u_h, u_numpy, _Gh_diag, degree_raise=degree_raise)
            if patch_cond_diag is not None:
                _save_patch_cond_hist(u_h, msh, loop_idx, results_dir, patch_cond_diag)
        # ---- Step 3: Cell-wise quantities -------------------------
        print("\n[3] Computing cell-wise quantities...")
        svd = compute_jacobian_svd(msh)

        # Aspect ratios (sigma_max / sigma_min) already in svd["AR"]
        ar = svd["AR"]
        max_ar_iter = float(np.max(ar))
        avg_ar_iter = float(np.mean(ar))
        iter_entry = {
            "loop_idx": loop_idx,
            "n_vertices": n_vertices_iter,
            "n_cells": n_cells_iter,
            "TRE": tre_iter,
            "max_aspect_ratio": max_ar_iter,
            "avg_aspect_ratio": avg_ar_iter,
        }
        iteration_metrics.append(iter_entry)
        print(f"  AR  max={max_ar_iter:.4e}  avg={avg_ar_iter:.4e}")
        if k == 1:
            G, eta_zz_fn = compute_G_tilde(u_h)
            print("  G_tilde (ZZ) computed")
        else:
            G, eta_zz_fn = compute_G_tilde_nz(u_h)
            print("  G_tilde (NZ/PPR) computed")

        eta_k, res1, omegas = compute_anisotropic_eta(u_h, f_rhs, G=G, k=k)
        eta_k_i = np.asarray(res1)[None, :] * np.asarray(omegas)
        print(f"  eta_k_i shape: {eta_k_i.shape}  (directions × cells)")

        # Per-iteration estimator metrics — stored in iter_entry so every loop row in the CSV has them
        eta_aniso  = float(np.sqrt(np.sum(np.asarray(eta_k))))
        gdim       = msh.geometry.dim
        eta_zz_val = float(np.sqrt(sum(np.sum(G[(i, i)]) for i in range(gdim))))
        iter_entry.update({
            "ERE_anisotropic": float(eta_aniso  / norm_grad_uh) if norm_grad_uh > 1e-30 else float('nan'),
            "EI_anisotropic":  float(eta_aniso  / norm_grad_e)  if norm_grad_e  > 1e-30 else float('nan'),
            "ERE_ZZ":          float(eta_zz_val / norm_grad_uh) if norm_grad_uh > 1e-30 else float('nan'),
            "EI_ZZ":           float(eta_zz_val / norm_grad_e)  if norm_grad_e  > 1e-30 else float('nan'),
            "eta_anisotropic": eta_aniso,
            "eta_ZZ":          eta_zz_val,
        })

        # ---- Step 4: Vertex-wise quantities -----------------------
        print("\n[4] Computing vertex-wise quantities...")
        G_P_arr, Q = compute_G_P(u_h, G)
        sigma_p   = compute_sigma_P(u_h, eta_k_i)
        lambda_p  = compute_lambda_P(u_h, svd)
        print(f"  G_P: {G_P_arr.shape},  Q: {Q.shape},  lambda_P: {lambda_p.shape}")

        # ---- Step 5: Equidistribution check & h update ------------
        print("\n[5] Checking equidistribution...")
        h_p, coarsen_any, refine_any = adapt_h(msh, eta_k_i, u_h, tol, lambda_p, alpha, correction_factor, sigma_p)
        h_p = np.clip(h_p, hmin, hmax)
        # Cap per-vertex anisotropy to prevent metric eigenvalue explosion.
        MAX_H_RATIO = 1e3  # limits λ_max/λ_min ≤ 1e6 in the metric
        h_min_per_vtx = np.max(h_p, axis=1, keepdims=True) / MAX_H_RATIO
        h_p = np.maximum(h_p, h_min_per_vtx)
        # ---- ISOTROPIC OVERRIDE (diagnostic) -------------------------
        # Force isotropic elements by taking the smallest prescribed
        # size across all 3 directions at each vertex.
        # Set FORCE_ISOTROPIC = True to activate.
        FORCE_ISOTROPIC = False
        if FORCE_ISOTROPIC:
            h_min_per_vertex = np.min(h_p, axis=1, keepdims=True)  # (n_verts, 1)
            h_p = np.broadcast_to(h_min_per_vertex, h_p.shape).copy()  # (n_verts, 3)
            print(f"  [ISO] Forced isotropic: h = min(h1,h2,h3) per vertex")
            print(f"  [ISO] h range: [{h_p.min():.4e}, {h_p.max():.4e}]")
        # ---------------------------------------------------------------## [ISOTROPIC TEST k=2] Force isotropic adaptation: broadcast the per-vertex minimum
        ## h across all directions so the metric is scalar (no anisotropy).
        #h_iso = np.min(h_p, axis=1, keepdims=True)          # (n_verts, 1) — smallest h per vertex
        #h_p   = np.broadcast_to(h_iso, h_p.shape).copy()    # same h in every direction
        #h_p   = np.clip(h_p, hmin, hmax)
        n_coarsen = int(np.sum(coarsen_any))
        n_refine  = int(np.sum(refine_any))

        # Cell-level: a cell is flagged if ANY of its 4 vertices is flagged
        ctv = msh.topology.connectivity(tdim, 0).array.reshape(-1, tdim + 1)
        n_cells_coarsen = int(np.sum(np.any(coarsen_any[ctv], axis=1)))
        n_cells_refine  = int(np.sum(np.any(refine_any[ctv], axis=1)))

        iter_entry["n_coarsen"]       = n_coarsen
        iter_entry["n_refine"]        = n_refine
        iter_entry["n_cells_coarsen"] = n_cells_coarsen
        iter_entry["n_cells_refine"]  = n_cells_refine
        print(
            f"  Vertices:  coarsen ≥1 dir: {n_coarsen}/{n_vertices_iter},  "
            f"refine ≥1 dir: {n_refine}/{n_vertices_iter}\n"
            f"  Cells:     coarsen ≥1 vtx: {n_cells_coarsen}/{n_cells_iter},  "
            f"refine ≥1 vtx: {n_cells_refine}/{n_cells_iter}"
        )

        # ---- Step 5b: Cap boundary anisotropy to surface curvature ----
        if tok_snap is not None:
            boundary_verts = get_boundary_vertex_indices(msh)
            bnd_cell_mask = np.any(np.isin(ctv, boundary_verts), axis=1)
            max_ar_bnd = float(np.max(ar[bnd_cell_mask])) if np.any(bnd_cell_mask) else float('nan')
            print(f"  Max AR at boundary cells: {max_ar_bnd:.4e}")
            for j in boundary_verts:
                h_sorted = np.sort(h_p[j])
                h_min_j = h_sorted[0]
                x = msh.geometry.x[j]
                R_local = np.sqrt(x[0]**2 + x[1]**2)
                if R_local < 1e-10:
                    continue
                h_max_tangential = np.sqrt(8.0 * R_local * h_min_j)
                h_p[j] = np.clip(h_p[j], h_min_j, h_max_tangential)

        # ---- Step 6: Build metric tensor --------------------------
        print("\n[6] Building metric tensor...")
        mesh_path = str(meshes_dir / f"mesh_{loop_idx}.mesh")
        sol_path  = str(sols_dir   / f"mesh_{loop_idx}.sol")
        perm = build_dolfinx_to_medit_map(msh)
        build_metric(h_p, Q, perm, mesh_path, sol_path)
        print(f"  Metric → {Path(sol_path).name}")

        # Attach u_h to every VTU. For k=2 (P2 elements) interpolate down to
        # CG1 so ParaView can read it as point data.
        if k > 1:
            from dolfinx import fem as _fem
            u_h_vtu = _fem.Function(_fem.functionspace(msh, ("CG", 1)))
            u_h_vtu.interpolate(u_h)
        else:
            u_h_vtu = u_h
        vtu_path = to_vtu(
            mesh_path,
            output_dir=vtk_dir,
            write_solution=True,
            u_h=u_h_vtu,
            dof_to_medit=perm,
        )
        if vtu_path:
            print(f"  VTU (+u_h) saved: {Path(vtu_path).name}")

        # ---- Step 7: MMG adaptation (skip on last iteration) ------
        if loop_idx < n_loop - 1:
            print("\n[7] Running MMG3D...")
            next_mesh_path = str(meshes_dir / f"mesh_{loop_idx + 1}.mesh")
            adapt_mesh_mmg(
                mesh_path, next_mesh_path, sol_path,
                mmg_log_file, mmg3d_exe,
                hgrad, hmin, hmax,
                vtk_dir=vtk_dir,
                extra_args=mmg_extra_args,
            )
            if tok_snap is not None:
                _snap_result = snap_tokamak_wall_vertices(
                    next_mesh_path,
                    r_inner=tok_snap["r_inner"],
                    r_outer=tok_snap["r_outer"],
                )
                _write_snap_row(loop_idx + 1, _snap_result)
            msh = read_medit_to_dolfinx(next_mesh_path)
            print(
                f"  New mesh: "
                f"{msh.topology.index_map(tdim).size_global} cells, "
                f"{msh.topology.index_map(0).size_global} vertices"
            )

        # ---- Last iteration: full checkpoint + final metrics ------
        if loop_idx == n_loop - 1:
            print("\n[SAVE] Storing computed quantities (adios4dolfinx)...")
            grad_uh = compute_gradient_dg0(u_h)
            save_computed_quantities(
                compute_errs_dir, u_h, grad_uh, eta_k, eta_zz_fn,
                time=float(loop_idx),
            )

            # eta_aniso / eta_zz_val already computed per-iteration above — reuse them
            final_metrics = {
                "n_vertices":      n_vertices_iter,
                "TRE":             tre_iter,
                "ERE_anisotropic": iter_entry["ERE_anisotropic"],
                "EI_anisotropic":  iter_entry["EI_anisotropic"],
                "ERE_ZZ":          iter_entry["ERE_ZZ"],
                "EI_ZZ":           iter_entry["EI_ZZ"],
                "eta_anisotropic": eta_aniso,
                "eta_ZZ":          eta_zz_val,
                "max_aspect_ratio": max_ar_iter,
                "avg_aspect_ratio": avg_ar_iter,
            }

            print("  Final metrics:")
            print(
                f"    n_vertices={n_vertices_iter}  TRE={tre_iter:.6e}"
                f"  ERE_aniso={final_metrics['ERE_anisotropic']:.6e}"
                f"  EI_aniso={final_metrics['EI_anisotropic']:.6e}"
            )
            print(
                f"    ERE_ZZ={final_metrics['ERE_ZZ']:.6e}"
                f"  EI_ZZ={final_metrics['EI_ZZ']:.6e}"
                f"  eta_aniso={eta_aniso:.6e}  eta_ZZ={eta_zz_val:.6e}"
            )

        print(f"\n[END] Iteration {loop_idx + 1} complete")

    print(f"\n{'='*70}")
    print("ADAPTIVE LOOP COMPLETE")
    print(f"{'='*70}")
    print(f"  Meshes      → {meshes_dir}")
    print(f"  Sol files   → {sols_dir}")
    print(f"  VTK files   → {vtk_dir}")
    print(f"  Quantities  → {compute_errs_dir}")
    print(f"  MMG log     → {mmg_log_file}")

    return msh, u_h, h_p, final_metrics, iteration_metrics
