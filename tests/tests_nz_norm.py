"""
PPR Diagnostic — isolate the source of EI_ZZ ≈ 4-5 for k=2.

Run this AFTER an adaptive iteration has produced u_h on an adapted mesh.
It computes four norms and prints a clear verdict:

  A = ||∇u_h - ∇u||        (true gradient error)
  B = ||∇u_h - G_nz||      (PPR estimator — what EI_ZZ uses)
  C = ||G_nz  - ∇u||       (recovery error — should be << A if superconvergent)
  D = ||∇u_h - Π_ZZ||      (ZZ estimator for comparison)

Expected if PPR is superconvergent:
  C << A  →  B ≈ A  →  EI = B/A ≈ 1

If instead C ≈ A:
  PPR is NOT superconvergent on this mesh  →  EI ≈ 2  (and can be higher)

If C > A:
  PPR is WORSE than the raw gradient  →  EI >> 1  (your 4-5x case)

Also tests on a UNIFORM mesh with the same PDE to check if the 4-5x
is mesh-dependent or inherent to the method.
"""

import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import fem
from dolfinx.fem import form
import ufl

# Make the repo-root modules importable when run from tests/
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def diagnose_ppr(u_h, u_numpy, Gh_func, degree_raise=3):
    """
    Parameters
    ----------
    u_h : fem.Function
        The P2 FE solution (from PDE solve or interpolation).
    u_numpy : callable
        Exact solution as numpy callback  u_numpy(x) -> array.
    Gh_func : callable
        The PPR recovery function:  Gh_func(u_h) -> fem.Function in P2 vector space.
    degree_raise : int
        Degree increment for the "truth" projection.
    """
    mesh = u_h.function_space.mesh
    gdim = mesh.geometry.dim
    degree = u_h.function_space.ufl_element().degree

    # --- Rich space for exact gradient proxy ---
    W = fem.functionspace(mesh, ("CG", degree + degree_raise))
    u_ex_W = fem.Function(W)
    u_ex_W.interpolate(u_numpy)
    u_h_W = fem.Function(W)
    u_h_W.interpolate(u_h)
    e_W = fem.Function(W)
    e_W.x.array[:] = u_ex_W.x.array - u_h_W.x.array

    # A = ||∇u_h - ∇u||  (true error)
    A_sq = fem.assemble_scalar(form(
        ufl.inner(ufl.grad(e_W), ufl.grad(e_W)) * ufl.dx))
    A = float(np.sqrt(A_sq))

    # --- PPR recovered gradient ---
    G_nz = Gh_func(u_h)

    # B = ||∇u_h - G_nz||  (PPR estimator)
    diff_B = ufl.grad(u_h) - G_nz
    B_sq = fem.assemble_scalar(form(ufl.inner(diff_B, diff_B) * ufl.dx))
    B = float(np.sqrt(B_sq))

    # C = ||G_nz - ∇u||  (recovery error)
    diff_C = G_nz - ufl.grad(u_ex_W)
    C_sq = fem.assemble_scalar(form(ufl.inner(diff_C, diff_C) * ufl.dx))
    C = float(np.sqrt(C_sq))

    # --- For comparison: ZZ estimator ---
    from eta_estimator1 import compute_zz_grad
    Pi_funcs = compute_zz_grad(u_h)
    diff_D = ufl.as_vector([ufl.grad(u_h)[i] - Pi_funcs[i] for i in range(gdim)])
    D_sq = fem.assemble_scalar(form(ufl.inner(diff_D, diff_D) * ufl.dx))
    D = float(np.sqrt(D_sq))

    # --- Print ---
    print(f"  A = ||∇u_h - ∇u||   = {A:.6e}   (true error)")
    print(f"  B = ||∇u_h - G_nz|| = {B:.6e}   (PPR estimator)")
    print(f"  C = ||G_nz  - ∇u||  = {C:.6e}   (recovery error)")
    print(f"  D = ||∇u_h - Π_ZZ|| = {D:.6e}   (ZZ estimator)")
    print()
    print(f"  EI_PPR = B/A = {B/A:.4f}   (should → 1)")
    print(f"  EI_ZZ  = D/A = {D/A:.4f}   (for comparison)")
    print(f"  C/A    = {C/A:.4f}   (< 1 means PPR is superconvergent)")
    print()

    if C / A < 0.3:
        print("  ✓ PPR IS superconvergent (C << A)")
        print("    → EI_PPR should be close to 1.")
        if B / A > 2.0:
            print("    ⚠ But EI_PPR is still large → check norm computation pipeline")
    elif C / A < 1.0:
        print("  ~ PPR is BETTER than raw gradient but not strongly superconvergent")
        print("    → EI_PPR will be > 1 but should improve with refinement")
    else:
        print("  ✗ PPR is NOT superconvergent on this mesh (C ≥ A)")
        print("    → EI_PPR >> 1 is expected — not a code bug")
        print("    → PPR needs more structured meshes for superconvergence with k=2")

    if D / A > 2.0 and degree > 1:
        print(f"\n  Note: ZZ estimator also overestimates (EI_ZZ={D/A:.2f}).")
        print("  This confirms ZZ is unsuitable for P2 — the DG0 projection")
        print("  of grad(u_h) loses the within-cell linear variation.")

    return {"A": A, "B": B, "C": C, "D": D}


def run_uniform_test(solver, u_numpy, Gh_func, N=8, k=2, degree_raise=3):
    """
    Run the same diagnostic on a UNIFORM mesh (no adaptation).
    If EI → 1 here but not on adapted meshes, the issue is
    PPR + adapted mesh, not a code bug.
    """
    print(f"\n{'='*60}")
    print(f"UNIFORM MESH TEST  (N={N}, k={k})")
    print(f"{'='*60}")

    mesh = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD, N, N, N,
        cell_type=dolfinx.mesh.CellType.tetrahedron,
    )
    u_h, f_rhs = solver(mesh)
    n_verts = mesh.topology.index_map(0).size_global
    n_cells = mesh.topology.index_map(3).size_global
    print(f"  {n_cells} cells, {n_verts} vertices")

    return diagnose_ppr(u_h, u_numpy, Gh_func, degree_raise)


def run_interpolation_test(u_numpy, Gh_func, N=8, k=2, degree_raise=3):
    """
    Run the diagnostic using INTERPOLATION instead of PDE solve.
    PPR is guaranteed to be superconvergent for interpolants of smooth
    functions on uniform meshes. If it fails here → code bug.
    """
    print(f"\n{'='*60}")
    print(f"INTERPOLATION TEST  (N={N}, k={k})")
    print(f"{'='*60}")

    mesh = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD, N, N, N,
        cell_type=dolfinx.mesh.CellType.tetrahedron,
    )
    V = fem.functionspace(mesh, ("CG", k))
    u_h = fem.Function(V)
    u_h.interpolate(u_numpy)
    n_verts = mesh.topology.index_map(0).size_global
    print(f"  {mesh.topology.index_map(3).size_global} cells, {n_verts} vertices")

    return diagnose_ppr(u_h, u_numpy, Gh_func, degree_raise)


if __name__ == "__main__":
    # ---- Adjust these imports to match your project ----
    from nz_eta_estimatorP2 import Gh
    from solver import solve_poisson, u_exact_np

    # Test 1: Interpolation on uniform mesh (must give EI ≈ 1)
    for N in (4, 8, 12):
        run_interpolation_test(u_exact_np, Gh, N=N, k=2)

    # Test 2: PDE solve on uniform mesh
    for N in (4, 8, 12):
        run_uniform_test(solve_poisson, u_exact_np, Gh, N=N, k=2)
