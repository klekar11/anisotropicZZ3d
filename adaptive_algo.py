# pyright: reportMissingImports=false, reportMissingModuleSource=false
import os
from pathlib import Path
from typing import Callable

import numpy as np

from solver import solve_poisson as _default_solver
from mesh_helpers import (
    write_cube_mesh,
    generate_initial_mesh,
    read_medit_to_dolfinx,
    to_vtu,
    build_dolfinx_to_medit_map,
    build_metric,
    adapt_mesh_mmg,
    save_computed_quantities,
)
from eta_estimator import (
    compute_jacobian_svd,
    adapt_h,
    compute_anisotropic_eta,
    compute_G_tilde,
    compute_G_P,
    compute_lambda_P,
    compute_sigma_P,
    compute_gradient_dg0,
)


def run_adaptive_poisson(
    results_dir: Path | str,
    solver: Callable | None = None,
    n_loop: int = 5,
    hmax: float = 1.0,
    hmin: float = 1e-10,
    hgrad: float = -1,
    tol: float = 0.5,
    alpha: float = 0.25,
    correction_factor: float = 1.5,
    mmg3d_exe: str = "/usr/local/bin/mmg3d_O3",
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
    mmg3d_exe :
        Path to the MMG3D executable.

    Returns
    -------
    msh :
        Final mesh (DOLFINx).
    u_h :
        Final FEM solution.
    h_p :
        Final target mesh-size array, shape ``(n_vertices, tdim)``.
    """
    if solver is None:
        solver = _default_solver

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
    # STEP 1: Generate initial mesh
    # ------------------------------------------------------------------
    print("\n[STEP 1] Generating initial mesh...")
    coarse_mesh_path = str(meshes_dir / "mesh_coarse.mesh")
    initial_mesh_path = str(meshes_dir / "mesh_0.mesh")

    write_cube_mesh(coarse_mesh_path)
    if not os.path.isfile(coarse_mesh_path):
        raise FileNotFoundError(f"Coarse mesh not created: {coarse_mesh_path}")

    generate_initial_mesh(coarse_mesh_path, initial_mesh_path, mmg3d_exe, hmax=hmax, hgrad=-1)
    if not os.path.isfile(initial_mesh_path):
        raise FileNotFoundError(f"Initial mesh not created by MMG: {initial_mesh_path}")

    msh = read_medit_to_dolfinx(initial_mesh_path)
    print(
        f"  Initial mesh: "
        f"{msh.topology.index_map(3).size_global} cells, "
        f"{msh.topology.index_map(0).size_global} vertices"
    )

    vtu_path = to_vtu(initial_mesh_path, output_dir=vtk_dir)
    if vtu_path:
        print(f"  VTU saved: {Path(vtu_path).name}")

    # ------------------------------------------------------------------
    # MAIN ADAPTIVE LOOP
    # ------------------------------------------------------------------
    u_h = None
    h_p = None

    for loop_idx in range(n_loop):
        print(f"\n{'='*70}")
        print(f"LOOP {loop_idx + 1}/{n_loop}")
        print(f"{'='*70}")

        # ---- Step 2: Solve ----------------------------------------
        print("\n[2] Solving Poisson equation...")
        u_h, f_rhs = solver(msh)
        tdim = msh.topology.dim
        print(
            f"  Solution on "
            f"{msh.topology.index_map(tdim).size_global} cells, "
            f"{msh.topology.index_map(0).size_global} vertices"
        )

        # ---- Step 3: Cell-wise quantities -------------------------
        print("\n[3] Computing cell-wise quantities...")
        svd = compute_jacobian_svd(msh)
        eta_k, res1, omegas = compute_anisotropic_eta(u_h, f_rhs)
        eta_k_i = np.asarray(res1)[None, :] * np.asarray(omegas)
        print(f"  eta_k_i shape: {eta_k_i.shape}  (directions × cells)")

        # compute_G_tilde now also returns the ZZ error as a fem.Function
        G, eta_zz_fn = compute_G_tilde(u_h)
        print("  G_tilde + eta_zz computed")

        # ---- Step 4: Vertex-wise quantities -----------------------
        print("\n[4] Computing vertex-wise quantities...")
        G_P_arr, Q = compute_G_P(u_h, G)
        sigma_p   = compute_sigma_P(u_h, eta_k_i)
        lambda_p  = compute_lambda_P(u_h, svd)
        print(f"  G_P: {G_P_arr.shape},  Q: {Q.shape},  lambda_P: {lambda_p.shape}")

        # ---- Step 5: Equidistribution check & h update ------------
        print("\n[5] Checking equidistribution...")
        h_p = adapt_h(msh, eta_k_i, u_h, tol, lambda_p, alpha, correction_factor, sigma_p)
        h_p = np.clip(h_p, hmin, hmax)

        # ---- Step 6: Build metric tensor --------------------------
        print("\n[6] Building metric tensor...")
        mesh_path = str(meshes_dir / f"mesh_{loop_idx}.mesh")
        sol_path  = str(sols_dir   / f"mesh_{loop_idx}.sol")
        perm = build_dolfinx_to_medit_map(msh)
        build_metric(h_p, Q, perm, mesh_path, sol_path)
        print(f"  Metric → {Path(sol_path).name}")

        # ---- Step 7: MMG adaptation (skip on last iteration) ------
        if loop_idx < n_loop - 1:
            print("\n[7] Running MMG3D...")
            next_mesh_path = str(meshes_dir / f"mesh_{loop_idx + 1}.mesh")
            adapt_mesh_mmg(
                mesh_path, next_mesh_path, sol_path,
                mmg_log_file, mmg3d_exe,
                hgrad, hmin, hmax,
                vtk_dir=vtk_dir,
            )
            msh = read_medit_to_dolfinx(next_mesh_path)
            print(
                f"  New mesh: "
                f"{msh.topology.index_map(tdim).size_global} cells, "
                f"{msh.topology.index_map(0).size_global} vertices"
            )

        # ---- Last iteration: checkpoint with adios4dolfinx --------
        if loop_idx == n_loop - 1:
            print("\n[SAVE] Storing computed quantities (adios4dolfinx)...")
            grad_uh = compute_gradient_dg0(u_h)
            save_computed_quantities(
                compute_errs_dir, u_h, grad_uh, eta_k, eta_zz_fn,
                time=float(loop_idx),
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

    return msh, u_h, h_p
