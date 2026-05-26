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
    :meth:`record` to store three L² gradient errors:

      A = ||∇u_h − ∇u||          true FE gradient error
      B = ||∇u_h − G_nz||        PPR–FE residual
      C = ||G_nz − ∇u||          PPR recovery error (vs. exact)

    Expected rates (P2, w.r.t. h_z in the boundary layer):
      A ~ O(h²),  B ~ O(h²),  C ~ O(h³) if superconvergent.

    The representative mesh size h_z is the median of ``h_p[:, 2]`` (the
    z-component of the anisotropic mesh-size array) over all vertices
    inside the boundary layer, defined by
    ``|x[layer_dir] − layer_centre| < layer_half_width``.

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
        Gh_func,
        grad_u_exact,
        h_p: "np.ndarray | None" = None,
    ) -> None:
        """Record one run's errors and representative h_z.

        Parameters
        ----------
        tol :
            Equidistribution tolerance for this run.
        msh :
            Converged DOLFINx mesh.
        u_h :
            Converged FEM solution (CG1 or CG2).
        Gh_func :
            Callable ``Gh_func(u_h) -> fem.Function`` returning the PPR
            recovered gradient (e.g. ``nz_eta_estimatorP2.Gh``).
        grad_u_exact :
            Callable compatible with ``fem.Function.interpolate`` on a
            vector space: ``x`` shape ``(gdim, n_dofs)`` → ``(gdim, n_dofs)``.
        h_p :
            Optional ``(n_vertices, tdim)`` array from :func:`adapt_h`.
            When provided ``h_p[:, 2]`` is used for h_z; otherwise h_z
            is estimated from each in-layer cell's z-vertex range.
        """
        from dolfinx import fem as _fem
        import ufl as _ufl
        from dolfinx.fem import form as _form

        gdim = msh.geometry.dim
        tdim = msh.topology.dim
        coords = msh.geometry.x          # (n_vertices, gdim)

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

        # ---- PPR recovered gradient (P2 vector fem.Function) ---------
        G_nz = Gh_func(u_h)

        # ---- exact gradient in a high-degree CG vector space ---------
        deg_ex = u_h.function_space.ufl_element().degree + 2
        V_grad = _fem.functionspace(msh, ("Lagrange", deg_ex, (gdim,)))
        grad_ex_fn = _fem.Function(V_grad, name="grad_u_exact")
        grad_ex_fn.interpolate(grad_u_exact)

        # ---- A = ||∇u_h − ∇u|| (true FE gradient error) -------------
        e_A = _ufl.grad(u_h) - grad_ex_fn
        A = float(np.sqrt(
            _fem.assemble_scalar(_form(_ufl.inner(e_A, e_A) * _ufl.dx))
        ))

        # ---- B = ||∇u_h − G_nz|| (PPR–FE residual) ------------------
        e_B = _ufl.grad(u_h) - G_nz
        B = float(np.sqrt(
            _fem.assemble_scalar(_form(_ufl.inner(e_B, e_B) * _ufl.dx))
        ))

        # ---- C = ||G_nz − ∇u|| (PPR recovery vs. exact) -------------
        e_C = G_nz - grad_ex_fn
        C = float(np.sqrt(
            _fem.assemble_scalar(_form(_ufl.inner(e_C, e_C) * _ufl.dx))
        ))

        entry = {"tol": tol, "h_layer": h_layer, "A": A, "B": B, "C": C}
        self._records.append(entry)
        print(
            f"  [PPRTracker] tol={tol:.3e}  h_z={h_layer:.3e}"
            f"  A={A:.3e}  B={B:.3e}  C={C:.3e}"
        )

    # ------------------------------------------------------------------
    def _sorted(self) -> list[dict]:
        return sorted(self._records, key=lambda r: r["h_layer"])

    def print_table(self) -> None:
        """Print a convergence table sorted by ascending h_z."""
        recs = self._sorted()
        header = (
            f"{'tol':>10}  {'h_z':>10}  {'A':>10}  {'B':>10}  {'C':>10}"
            f"  {'rate_A':>7}  {'rate_B':>7}  {'rate_C':>7}"
        )
        sep = "─" * len(header)
        print("\n" + header)
        print(sep)
        for k, r in enumerate(recs):
            if k == 0:
                rate_str = f"{'—':>7}  {'—':>7}  {'—':>7}"
            else:
                prev = recs[k - 1]
                log_h = np.log(r["h_layer"] / prev["h_layer"])
                ra = np.log(r["A"] / prev["A"]) / log_h
                rb = np.log(r["B"] / prev["B"]) / log_h
                rc = np.log(r["C"] / prev["C"]) / log_h
                rate_str = f"{ra:7.2f}  {rb:7.2f}  {rc:7.2f}"
            print(
                f"  {r['tol']:>8.3e}  {r['h_layer']:>10.3e}"
                f"  {r['A']:>10.3e}  {r['B']:>10.3e}  {r['C']:>10.3e}"
                f"  {rate_str}"
            )

    def plot(self, filename: str = "ppr_convergence.pdf") -> None:
        """Save a log–log convergence plot to *filename*."""
        import matplotlib.pyplot as plt

        recs = self._sorted()
        if len(recs) == 0:
            print("  [PPRTracker] No records yet — nothing to plot.")
            return

        h = np.array([r["h_layer"] for r in recs])
        A = np.array([r["A"] for r in recs])
        B = np.array([r["B"] for r in recs])
        C = np.array([r["C"] for r in recs])

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.loglog(h, A, "b-o", label=r"$A=\|\nabla u_h-\nabla u\|$")
        ax.loglog(h, B, "r-s", label=r"$B=\|\nabla u_h-G_{nz}\|$")
        ax.loglog(h, C, "g-^", label=r"$C=\|G_{nz}-\nabla u\|$")

        # Reference slopes — only drawn when there are at least 2 points
        if len(recs) >= 2:
            i_mid = len(recs) // 2
            h0 = h[i_mid]
            ax.loglog(
                h, A[i_mid] * (h / h0) ** 2,
                "b--", alpha=0.45, lw=1.2, label=r"$O(h^2)$",
            )
            ax.loglog(
                h, C[i_mid] * (h / h0) ** 3,
                "g--", alpha=0.45, lw=1.2, label=r"$O(h^3)$",
            )

        ax.set_xlabel(r"$h_z$ (median in boundary layer)")
        ax.set_ylabel(r"$L^2$ gradient error")
        ax.set_title("PPR gradient-recovery convergence")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()

        out = Path(filename).resolve()
        fig.savefig(str(out), dpi=150)
        # Also save a PNG alongside the PDF for quick viewing
        png_out = out.with_suffix(".png")
        fig.savefig(str(png_out), dpi=150)
        plt.close(fig)
        print(f"  [PPRTracker] Convergence plot saved → {out}")
        print(f"  [PPRTracker]                    PNG → {png_out}")


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
    degree_raise: int = 2,
    mmg3d_exe: str = "/usr/local/bin/mmg3d_O3",
    initial_mesh_file: Path | str | None = None,
    k: int = 1,
    mmg_extra_args: list | None = None,
    tok_snap: dict | None = None,
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
        print(f"  TRE={tre_iter:.6e}  n_vertices={n_vertices_iter}")
        if not np.isfinite(tre_iter):
            print("  WARNING: TRE is NaN/inf — solution is degenerate, stopping loop.")
            break
# ---- Step 2c: PPR diagnostic (remove after debugging) -----
        if k == 2:
            from nz_eta_estimatorP2 import Gh as _Gh_diag
            from diagnose_ppr import diagnose_ppr
            print(f"\n[2c] PPR diagnostic (loop {loop_idx}):")
            diagnose_ppr(u_h, u_numpy, _Gh_diag, degree_raise=degree_raise)
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
        # Without this, h_p → hmin in one direction gives λ = 1/hmin² → MMG3D failure.
        MAX_H_RATIO = 1e3  # limits λ_max/λ_min ≤ 1e6 in the metric
        h_min_per_vtx = np.max(h_p, axis=1, keepdims=True) / MAX_H_RATIO
        h_p = np.maximum(h_p, h_min_per_vtx)
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

            # Reuse norms computed in the per-iteration block above
            eta_aniso = float(np.sqrt(np.sum(np.asarray(eta_k))))
            gdim = msh.geometry.dim
            eta_zz_val = float(
                np.sqrt(sum(np.sum(G[(i, i)]) for i in range(gdim)))
            )

            final_metrics = {
                "n_vertices": n_vertices_iter,
                "TRE": tre_iter,
                "ERE_anisotropic": float(eta_aniso / norm_grad_uh),
                "EI_anisotropic": float(eta_aniso / norm_grad_e),
                "ERE_ZZ": float(eta_zz_val / norm_grad_uh),
                "EI_ZZ": float(eta_zz_val / norm_grad_e),
                "eta_anisotropic": eta_aniso,
                "eta_ZZ": eta_zz_val,
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
