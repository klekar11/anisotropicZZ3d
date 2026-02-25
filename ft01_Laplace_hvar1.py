from dolfinx import default_scalar_type
from dolfinx.fem import (
    Expression,
    Function,
    functionspace,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_cube, locate_entities_boundary

from mpi4py import MPI
from ufl import (
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    div,
    dot,
    dx,
    grad,
    inner,
)

import ufl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def u_ex(mod):
    return lambda x: mod.sin(mod.pi * x[0]) * mod.sin(2 * mod.pi * x[1]) * mod.sin(mod.pi * x[2])

u_numpy = u_ex(np)
u_ufl = u_ex(ufl)

def solve_poisson(N=8, degree=1):
    mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N)
    x = SpatialCoordinate(mesh)
    f = -div(grad(u_ufl(x)))
    V = functionspace(mesh, ("Lagrange", degree))
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    L = f * v * dx
    facets = locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda x: np.full(x.shape[1], True)
    )
    dofs = locate_dofs_topological(V, mesh.topology.dim - 1, facets)
    bcs = [dirichletbc(default_scalar_type(0), dofs, V)]
    default_problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="poisson_convergence_",
    )
    return default_problem.solve(), u_ufl(x)

def errors_computation(uh, u_ex, degree_raise=3):
    # Create higher order function space
    degree = uh.function_space.ufl_element().degree
    family = uh.function_space.ufl_element().family_name
    mesh = uh.function_space.mesh
    W = functionspace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    u_W = Function(W)
    u_W.interpolate(uh)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = Function(W)
    if isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = Expression(u_ex, W.element.interpolation_points)
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)

    # Compute the error in the higher order function space
    e_W = Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    # Integrate the L2 error
    error_l2 = form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local_l2 = assemble_scalar(error_l2)
    error_global_l2 = mesh.comm.allreduce(error_local_l2, op=MPI.SUM)
    
    error_grad = form(dot(grad(e_W), grad(e_W)) * dx)
    error_local_grad = assemble_scalar(error_grad)
    error_global_grad = mesh.comm.allreduce(error_local_grad, op=MPI.SUM)
    
    error_max_local = np.max(np.abs(e_W.x.array))
    error_max = mesh.comm.allreduce(error_max_local, op=MPI.MAX)
    
    return np.sqrt(error_global_l2), np.sqrt(error_global_grad), error_max
    

Ns = [4, 8, 16, 32]
EsL2 = np.zeros(len(Ns), dtype=default_scalar_type)
EsGrad = np.zeros(len(Ns), dtype=default_scalar_type)
EsInf = np.zeros(len(Ns), dtype=default_scalar_type)
hs = np.zeros(len(Ns), dtype=np.float64)
for i, N in enumerate(Ns):
    uh, u_ex_ufl = solve_poisson(N, degree=1)
    comm = uh.function_space.mesh.comm
    # One can send in either u_numpy or u_ex
    # For L2 error estimations it is reccommended to send in u_numpy
    # as no JIT compilation is required
    EsL2[i], EsGrad[i], EsInf[i] = errors_computation(uh, u_numpy)
    hs[i] = 1.0 / Ns[i]
    if comm.rank == 0:
        print(f"h: {hs[i]:.2e} L2-Error: {EsL2[i]:.2e} H_0-Error: {EsGrad[i]:.2e} Infinity-Error: {EsInf[i]:.2e}")

# Plotting

if comm.rank == 0:

    h_tick_labels = [f"2^{int(round(np.log2(h)))}" for h in hs]

    plots = [
        (EsL2,   "L2 Error",         hs**2, "h²",  "steelblue"),
        (EsGrad, "H1 Seminorm Error", hs**1, "h¹",  "seagreen"),
        (EsInf,  "Linf Error",        hs**2, "h²",  "tomato"),
    ]

    for errors, name, theo, theo_label, color in plots:

        # Scale theoretical line to start at same level as computed error
        theo_scaled = theo * (errors[0] / theo[0])

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.loglog(hs, errors,      marker="o", color=color, label="Computed error")
        ax.loglog(hs, theo_scaled, "--",        color="gray", label=theo_label)

        ax.set_xlabel("h")
        ax.set_ylabel("Error")
        ax.set_xticks(hs)
        ax.set_xticklabels(h_tick_labels)
        ax.invert_xaxis()
        ax.legend()
        ax.grid(True, which="both", linestyle=":")
        plt.tight_layout()
        plt.savefig(f"convergence_{name.replace(' ', '_')}.png", dpi=150)
        plt.close()

