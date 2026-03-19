# pyright: reportMissingImports=false, reportMissingModuleSource=false
import sys
import os
from pathlib import Path
import numpy as np
import meshio
# Add parent directory to path for imports
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from solver import solve_poisson
from mesh_helpers import (
    write_cube_mesh,
    generate_initial_mesh,
    read_medit_to_dolfinx,
    to_vtu,
    build_dolfinx_to_medit_map,
    build_metric,
    adapt_mesh_mmg,
)

from eta_estimator import (
    compute_jacobian_svd,
    adapt_h,
    compute_anisotropic_eta,
    compute_G_tilde,
    compute_G_P,
    compute_lambda_P,
    compute_sigma_P,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_LOOP = 5
HMAX = 1
HMIN = 1e-10
HGRAD = -1
TOL = 0.5
ALPHA = 0.25
MMG3D_EXE = "/usr/local/bin/mmg3d_O3"  # or "mmg3d_O3" depending on installation
corrective_param = 1.5
# ---------------------------------------------------------------------------
# Create results directory
# ---------------------------------------------------------------------------
results_dir = Path(__file__).resolve().parent / "results"
results_dir.mkdir(exist_ok=True, parents=True)
mmg_log_file = results_dir / "mmg_output.txt"
OUTPUT_DIR = str(results_dir)

print(f"[INIT] Results directory: {results_dir}")

# ==========================================================================
# STEP 1: Generate initial mesh
# ==========================================================================
print("\n[STEP 1] Generating initial mesh...")
p = lambda name: os.path.join(OUTPUT_DIR, name)
coarse_mesh_path = p("mesh_coarse.mesh")
initial_mesh_path = p("mesh_0.mesh")
write_cube_mesh(coarse_mesh_path)

if not os.path.isfile(coarse_mesh_path):
    raise FileNotFoundError(
        f"Coarse mesh file was not created: {coarse_mesh_path}"
    )

generate_initial_mesh(coarse_mesh_path, initial_mesh_path, MMG3D_EXE, hmax=HMAX, hgrad=-1)

if not os.path.isfile(initial_mesh_path):
    raise FileNotFoundError(
        f"Initial mesh file was not created by MMG: {initial_mesh_path}"
    )

msh = read_medit_to_dolfinx(initial_mesh_path)
n_cells = msh.topology.index_map(3).size_global
n_verts = msh.topology.index_map(0).size_global
print(f"  Initial mesh loaded: {n_cells} cells, {n_verts} vertices")

vtu_path = to_vtu(initial_mesh_path)
if vtu_path:
    print(f"  VTU saved: {Path(vtu_path).name}")

# ==========================================================================
# MAIN ADAPTIVE LOOP
# ==========================================================================
for loop_idx in range(N_LOOP):
    print(f"\n{'='*70}")
    print(f"LOOP {loop_idx + 1}/{N_LOOP}")
    print(f"{'='*70}")

    # ----------------------------------------------------------------------
    # STEP 2: Solve Poisson
    # ----------------------------------------------------------------------
    print("\n[2] Solving Poisson equation...")
    u_h, f_rhs = solve_poisson(msh, degree=1)
    tdim = msh.topology.dim
    gdim = msh.geometry.dim
    n_cells = msh.topology.index_map(tdim).size_global
    n_verts = msh.topology.index_map(0).size_global
    print(f"  Solution computed on {n_cells} cells, {n_verts} vertices")

    # ----------------------------------------------------------------------
    # STEP 3: Cell-wise quantities (no loop needed, vectorised)
    # ----------------------------------------------------------------------
    print("\n[3] Computing cell-wise quantities...")

    svd = compute_jacobian_svd(msh)
    print("  Jacobian SVD computed")
    eta_k, res1, omegas = compute_anisotropic_eta(u_h, f_rhs)

    eta_k_i = np.asarray(res1)[None, :] * np.asarray(omegas)
    print(f"  eta_k_i shape: {eta_k_i.shape}  (directions x cells)")

    G = compute_G_tilde(u_h)
    print("  G_tilde computed")

    # ----------------------------------------------------------------------
    # STEP 4: Vertex-wise quantities
    # ----------------------------------------------------------------------
    print("\n[4] Computing vertex-wise quantities...")

    G_P_arr, Q = compute_G_P(u_h, G)
    print(f"  G_P shape: {G_P_arr.shape},  Q shape: {Q.shape}")
    
    sigma_p = compute_sigma_P(u_h, eta_k_i)
    print(f"  sigma_P shape: {sigma_p.shape}")

    lambda_p = compute_lambda_P(u_h, svd)
    print(f"  lambda_P shape: {lambda_p.shape}")

    # ----------------------------------------------------------------------
    # STEP 5: Check equidistribution and adjust mesh sizes
    # ----------------------------------------------------------------------
    print("\n[5] Checking equidistribution conditions...")
    h_p = adapt_h(msh, eta_k_i, u_h, TOL, lambda_p, ALPHA, sigma_p, corrective_param)

    # Clamp h values to [HMIN, HMAX]
    h_p = np.clip(h_p, HMIN, HMAX)

    # ----------------------------------------------------------------------
    # STEP 6: Build metric tensor and write .sol
    # ----------------------------------------------------------------------
    print("\n[6] Building metric tensor...")

    mesh_path = str(results_dir / f"mesh_{loop_idx}.mesh")
    sol_path  = str(results_dir / f"mesh_{loop_idx}.sol")
    perm = build_dolfinx_to_medit_map(msh)

    build_metric(h_p, Q, perm, mesh_path, sol_path)
    print(f"  Metric written to mesh_{loop_idx}.sol")

    # ----------------------------------------------------------------------
    # STEP 7: Run MMG3D to adapt the mesh
    # ----------------------------------------------------------------------
    if loop_idx < N_LOOP - 1:
        print("\n[7] Running MMG3D...")
        next_mesh_path = str(results_dir / f"mesh_{loop_idx + 1}.mesh")

        adapt_mesh_mmg(
            mesh_path,
            next_mesh_path,
            sol_path,
            mmg_log_file,
            MMG3D_EXE,
            HGRAD,
            HMIN,
            HMAX,
        )

        # Load the adapted mesh for next iteration
        msh = read_medit_to_dolfinx(next_mesh_path)
        n_cells = msh.topology.index_map(tdim).size_global
        n_verts = msh.topology.index_map(0).size_global
        print(f"  New mesh loaded: {n_cells} cells, {n_verts} vertices")

    print(f"\n[END] Iteration {loop_idx + 1} complete")

# ==========================================================================
print(f"\n{'='*70}")
print("ADAPTIVE LOOP COMPLETE")
print(f"{'='*70}")
print(f"Results stored in: {results_dir}")
print(f"MMG output log:    {mmg_log_file}")