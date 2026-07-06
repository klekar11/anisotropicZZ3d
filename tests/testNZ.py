"""
Polynomial-preserving property test for the 3D Naga-Zhang / PPR recovery.

Theory: if u ∈ P3, then G_h(I_h u) = ∇u  exactly (up to machine precision).
We use u(x,y,z) = x³ + y³ + z³, so ∇u = (3x², 3y², 3z²).

The test interpolates u into the P2 FE space (which cannot represent a cubic
exactly — that's fine, PPR fits a cubic patch and recovers the derivative of
the *true* cubic from the P2 samples), then calls Gh and compares the result
at every DOF of the P2-vector space to the analytical gradient.

If the max pointwise error is O(1e-10) or smaller → the recovery is correct.
If it is O(1)                                     → the old parallel/cached
                                                     version is still running.
If it is ≈ 0.5 * ∇u                               → race condition (only one
                                                     of the two 0.5-contributions
                                                     reaches each mid-edge DOF).
"""

import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

import dolfinx
from dolfinx import fem

# Make the repo-root modules importable when run from tests/
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---- import YOUR existing Gh ------------------------------------------
from nz_eta_estimatorP2 import Gh


def test_polynomial_preserving(N: int = 6):
    """Run the test on an N×N×N unit-cube tet mesh."""

    mesh = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD, N, N, N,
        cell_type=dolfinx.mesh.CellType.tetrahedron,
    )

    # P2 scalar space
    V = fem.functionspace(mesh, ("CG", 2))
    uh = fem.Function(V)
    uh.interpolate(lambda x: x[0]**3 + x[1]**3 + x[2]**3)

    # Recovered gradient (should be a P2 vector function)
    Ghuh = Gh(uh)

    # Exact gradient evaluated at the DOF coordinates of the P2-vector space
    V2 = Ghuh.function_space
    dof_coords = V2.tabulate_dof_coordinates()  # (n_dofs, 3)

    grad_exact = np.zeros_like(Ghuh.x.array)
    n_dofs_scalar = dof_coords.shape[0]

    # V2 is ("Lagrange", 2, (3,)).  DOF layout: the first n_dofs_scalar
    # entries are component 0, the next are component 1, etc.
    # This is the default block layout in dolfinx for vector spaces.
    bs = V2.dofmap.bs                         # block size = 3
    n_blocks = V2.dofmap.index_map.size_local  # = n_dofs_scalar

    for block in range(n_blocks):
        x, y, z = dof_coords[block]
        grad_exact[block * bs + 0] = 3.0 * x**2
        grad_exact[block * bs + 1] = 3.0 * y**2
        grad_exact[block * bs + 2] = 3.0 * z**2

    # Compare
    err = Ghuh.x.array[:len(grad_exact)] - grad_exact
    max_err = np.max(np.abs(err))
    l2_err  = np.sqrt(np.sum(err**2))
    l2_exact = np.sqrt(np.sum(grad_exact**2))
    rel_err = l2_err / l2_exact if l2_exact > 0 else l2_err

    print(f"Polynomial-preserving test  (N={N}, mesh {N}×{N}×{N})")
    print(f"  u(x) = x³ + y³ + z³")
    print(f"  ∇u   = (3x², 3y², 3z²)")
    print(f"  n_dofs(V2) = {n_blocks} blocks × {bs} components = {n_blocks*bs}")
    print(f"  max |G_h u_h - ∇u|  = {max_err:.6e}")
    print(f"  rel L2 error         = {rel_err:.6e}")

    if max_err < 1e-8:
        print("  ✓ PASS — polynomial-preserving property holds")
    elif rel_err > 0.3:
        print("  ✗ FAIL — large error; likely race condition or stale numba cache")
        print("    Try:  rm -rf __pycache__ *.nbi *.nbc")
    else:
        print(f"  ? MARGINAL — error {max_err:.2e} is above machine eps but below O(1)")
        print("    Check boundary patches and patch conditioning")

    # Detailed breakdown: vertex DOFs vs mid-edge DOFs
    n_vertices = mesh.topology.index_map(0).size_local
    # In a P2 scalar space, DOFs 0..n_vertices-1 sit on vertices,
    # the rest are mid-edge DOFs.  The vector space interleaves them.
    vertex_err = []
    edge_err = []
    for block in range(n_blocks):
        e = np.max(np.abs(err[block*bs : block*bs + bs]))
        if block < n_vertices:
            vertex_err.append(e)
        else:
            edge_err.append(e)

    vertex_err = np.array(vertex_err) if vertex_err else np.array([0.0])
    edge_err   = np.array(edge_err)   if edge_err   else np.array([0.0])
    print(f"\n  Vertex DOFs:   max err = {np.max(vertex_err):.6e}  "
          f"(n={len(vertex_err)})")
    print(f"  Mid-edge DOFs: max err = {np.max(edge_err):.6e}  "
          f"(n={len(edge_err)})")

    if len(edge_err) > 0 and np.max(edge_err) > 100 * np.max(vertex_err):
        print("  ⚠ Mid-edge error >> vertex error → "
              "likely only one 0.5-contribution landed (race condition)")

    return max_err


if __name__ == "__main__":
    # Run on a few mesh sizes to confirm it's not mesh-dependent
    for N in (4, 6, 8):
        test_polynomial_preserving(N)
        print()
