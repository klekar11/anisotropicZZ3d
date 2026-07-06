import numpy as np
import ufl
from ufl import SpatialCoordinate, TestFunction, TrialFunction, grad, inner, dx, div
from mpi4py import MPI
import dolfinx
from dolfinx import fem
from dolfinx.fem import functionspace, Function, dirichletbc, locate_dofs_topological, Constant
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import exterior_facet_indices
from typing import Callable

EPSILON = 0.01
SPHERE_R = 0.5

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


#def u_exact_sphere_np(
#    x: np.ndarray,
#    R: float = SPHERE_R,
#    epsilon: float = EPSILON,
#) -> np.ndarray:
#
#    r = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
#    return 0.5 * (1.0 + np.tanh((R - r) / epsilon))

def u_exact_sphere_np(
    x: np.ndarray,
    R: float = SPHERE_R,
    epsilon: float = EPSILON,
) -> np.ndarray:
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    s = R - r                              # signed dist: >0 inside, <0 outside
    result = np.where(
        s >=  epsilon, 1.0,
        np.where(
            s <= -epsilon, 0.0,
            (s + epsilon) / (2.0 * epsilon)
            + np.sin(np.pi * s / epsilon) / (2.0 * np.pi)
        )
    )
    return result

def solve_sphere_poisson(
    msh,
    degree: int = 1,
    R: float = SPHERE_R,
    epsilon: float = EPSILON,
):
    x   = SpatialCoordinate(msh)
    r   = ufl.sqrt(x[0]**2 + x[1]**2 + x[2]**2 + 1e-16)  # avoids grad singularity at 0
    s   = R - r                    # signed distance: positive inside, negative outside

    # --- Exact solution in UFL (for the BC interpolation expression) ---
    in_layer  = ufl.And(ufl.ge(s, -epsilon), ufl.le(s, epsilon))
    u_ex_ufl  = ufl.conditional(
        ufl.ge(s,  epsilon), 1.0,
        ufl.conditional(
            ufl.le(s, -epsilon), 0.0,
            (s + epsilon) / (2.0 * epsilon)
            + ufl.sin(ufl.pi * s / epsilon) / (2.0 * ufl.pi)
        )
    )

    # --- Manufactured RHS: f = -Δu_ex, computed analytically ---
    # In the transition zone:  Δu = d²u/ds² * |∇s|² + du/ds * Δs
    #   |∇s|² = 1  (s = R - r, |∇r| = 1)
    #   Δs    = -Δr = -2/r  (3D Laplacian of r)
    #   du/ds  = 1/(2ε) * (1 + cos(πs/ε))
    #   d²u/ds² = -π/(2ε²) * sin(πs/ε)
    # Outside the layer: f = 0 exactly (u is a constant there)
    f = ufl.conditional(
        in_layer,
        ufl.pi / (2.0 * epsilon**2) * ufl.sin(ufl.pi * s / epsilon)
        + (1.0 / (epsilon * r)) * (1.0 + ufl.cos(ufl.pi * s / epsilon)),
        ufl.as_ufl(0.0)
    )

    # --- Variational problem ---
    V  = functionspace(msh, ("Lagrange", degree))
    u  = TrialFunction(V)
    v  = TestFunction(V)
    a  = inner(grad(u), grad(v)) * dx
    L  = f * v * dx

    # Non-homogeneous Dirichlet BCs: u_h = u_ex on ∂Ω
    # (≈ 0 on the cube boundary since R+ε = 0.65 < 1)
    u_bc = Function(V)
    u_bc.interpolate(lambda x: u_exact_sphere_np(x, R=R, epsilon=epsilon))

    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = exterior_facet_indices(msh.topology)
    dofs = locate_dofs_topological(V, fdim, boundary_facets)
    bcs  = [dirichletbc(u_bc, dofs)]

    # --- Linear solver: CG + AMG — correct choice for SPD Poisson ---
    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": "cg",
            "pc_type":  "hypre",
            "pc_hypre_type": "boomeramg",
        },
        petsc_options_prefix="sphere_",
    )
    u_h = problem.solve()
    return u_h, f
#def solve_sphere_poisson(
#    R: float = SPHERE_R,
#    epsilon: float = EPSILON,
#):
#    x = SpatialCoordinate(msh)
#
#    # Offset avoids the singular derivative of sqrt at r=0.
#    r = ufl.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + 1e-16)
#    u_ex_ufl = 0.5 * (1.0 + ufl.tanh((R - r) / epsilon))
#    f = -ufl.div(ufl.grad(u_ex_ufl))
#
#    V = functionspace(msh, ("Lagrange", degree))
#    u = TrialFunction(V)
#    v = TestFunction(V)
#    a = inner(grad(u), grad(v)) * dx
#    L = f * v * dx
#
#    u_bc = Function(V)
#    u_bc.interpolate(lambda x: u_exact_sphere_np(x, R=R, epsilon=epsilon))
#
#    tdim = msh.topology.dim
#    fdim = tdim - 1
#    msh.topology.create_connectivity(fdim, tdim)
#    boundary_facets = exterior_facet_indices(msh.topology)
#    dofs = locate_dofs_topological(V, fdim, boundary_facets)
#    bcs = [dirichletbc(u_bc, dofs)]
#
#    problem = LinearProblem(
#        a,
#        L,
#        bcs=bcs,
#        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "pc_hypre_type": "boomeramg"},
#        petsc_options_prefix="sphere_",
#    )
#    u_h = problem.solve()
#    return u_h, f


# # TCV cross-section center and inner torus radius (in mm)
# R0 = 880.0     # major radius (center of cross-section in R)
# Z0 = 400.0     # vertical center of cross-section
# R_INNER = 200.0 # radius of the inner circle in the (R,Z) cross-section


def _u_exact_numpy(x, epsilon, R0, Z0, r0):
    R = np.sqrt(x[0]**2 + x[1]**2)
    d = np.sqrt((R - R0)**2 + (x[2] - Z0)**2)
    return 0.5 * (1.0 + np.tanh((r0 - d) / epsilon))


def solve_poisson_torus(msh, R0, Z0, r0, degree=1, epsilon=20.0):
    x = SpatialCoordinate(msh)

    # Break down the expression into intermediate UFL variables
    R = ufl.sqrt(x[0]**2 + x[1]**2)
    dR = R - Constant(msh, np.float64(R0))
    dZ = x[2] - Constant(msh, np.float64(Z0))
    d_sq = dR**2 + dZ**2
    d = ufl.sqrt(d_sq)

    eps_c = Constant(msh, np.float64(epsilon))
    r0_c = Constant(msh, np.float64(r0))
    arg = (r0_c - d) / eps_c

    u_exact = 0.5 * (1.0 + ufl.tanh(arg))

    # Compute f = -div(grad(u_exact)) symbolically but with broken-down tree
    f = -div(grad(u_exact))

    V = functionspace(msh, ("Lagrange", degree))
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx

    # Interpolate f into a Function to avoid expensive quadrature
    from dolfinx.fem import Expression
    f_func = Function(V)
    f_expr = Expression(f, V.element.interpolation_points)
    f_func.interpolate(f_expr)

    L = f_func * v * dx

    # Dirichlet BC from exact solution
    u_bc = Function(V)
    u_bc.interpolate(lambda x: _u_exact_numpy(x, epsilon, R0, Z0, r0))

    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = exterior_facet_indices(msh.topology)
    dofs = locate_dofs_topological(V, fdim, boundary_facets)
    bcs = [dirichletbc(u_bc, dofs)]

    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "pc_hypre_type": "boomeramg"},
        petsc_options_prefix="poisson_torus_",
    )
    u_h = problem.solve()
    return u_h, f_func


def solve_poisson_generic(
    msh,
    f,
    g: Callable[[np.ndarray], np.ndarray],
    k: int = 1,
    quadrature_degree: int | None = None,
) -> tuple:
    """Solve -Δu = f, u|∂Ω = g using Pk finite elements.

    Parameters
    ----------
    msh               : dolfinx.mesh.Mesh
    f                 : UFL expression or numpy callable for the RHS.
                        If a numpy callable, it is interpolated into a CG-k Function.
    g                 : numpy callable for the Dirichlet BC.
    k                 : polynomial degree (1 or 2).
    quadrature_degree : override the default quadrature degree used in assembly.
                        Pass e.g. 2*k+6 for smooth but sharply-peaked RHS (tok-sphere)
                        so that f is evaluated at Gauss points rather than pre-interpolated.
                        None → let FEniCSx choose automatically.

    Returns
    -------
    (u_h, f_coeff) where f_coeff is UFL-compatible (used by eta estimator).
    """
    V = fem.functionspace(msh, ("CG", k))

    uD = fem.Function(V)
    uD.interpolate(g)

    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    if isinstance(f, ufl.core.expr.Expr):
        # Use the UFL expression directly so it is evaluated at Gauss quadrature
        # points during assembly. Pre-interpolating onto CG-k nodal points loses
        # thin-layer sources (e.g. sphere ε=0.05) when no vertex falls in the layer.
        f_coeff = f
    else:
        # numpy callable or already a fem.Function
        F = fem.Function(V)
        F.interpolate(f)
        f_coeff = F

    dx_meta = (
        ufl.dx(metadata={"quadrature_degree": quadrature_degree})
        if quadrature_degree is not None
        else ufl.dx
    )
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * dx_meta
    L = f_coeff * v * dx_meta

    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "pc_hypre_type": "boomeramg"},
        petsc_options_prefix="poisson_generic_",
    )
    u_h = problem.solve()
    return u_h, f_coeff














































