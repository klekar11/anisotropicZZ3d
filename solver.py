import numpy as np
import ufl
from ufl import SpatialCoordinate, TestFunction, TrialFunction, grad, inner, dx
from mpi4py import MPI
from dolfinx import fem
from dolfinx.fem import functionspace, Function, dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import exterior_facet_indices

EPSILON = 0.01

def u_exact_np(x: np.ndarray) -> np.ndarray:

    return np.tanh(x[2] / EPSILON)


def solve_poisson(msh, degree: int = 1, epsilon: float = 0.01):
    x = SpatialCoordinate(msh)
    t = ufl.tanh(x[2] / epsilon)
    f = (2.0 / epsilon**2) * t * (1.0 - t**2)

    V = functionspace(msh, ("Lagrange", degree))
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    L = f * v * dx

    u_bc = Function(V)
    u_bc.interpolate(lambda x: np.tanh(x[2] / epsilon))

    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = exterior_facet_indices(msh.topology)
    dofs = locate_dofs_topological(V, fdim, boundary_facets)
    bcs = [dirichletbc(u_bc, dofs)]

    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "gmres", "pc_type": "ilu"},
        petsc_options_prefix="poisson_",
    )
    u_h = problem.solve()
    return u_h, f