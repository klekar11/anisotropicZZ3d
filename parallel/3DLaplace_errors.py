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
from estimator import compute_eta, compute_gradient_dg0, compute_zz_grad, compute_eta_zz, compute_G_tilde, get_G_matrix
from error_metrics import compute_error_metrics, compute_error_metrics_ZZ

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
        petsc_options={"ksp_type": "gmres", "pc_type": "none"},
        petsc_options_prefix="poisson_convergence_",
    )
    return default_problem.solve(), u_ufl(x)

Ns = [4, 8, 16, 32]
hs = np.zeros(len(Ns), dtype=np.float64)


TREs    = []
EREs    = []
EIs     = []
ERE_ZZs = []
EI_ZZs  = []

for i, N in enumerate(Ns):
    uh, u_ex_ufl = solve_poisson(N, degree=1)
    domain = uh.function_space.mesh
    comm = domain.comm
    V = uh.function_space
    x = SpatialCoordinate(domain)
    f = -div(grad(u_ufl(x)))

    u_ex_func = Function(V)
    u_ex_func.interpolate(u_numpy)

    all_eta = compute_eta(uh, f)
    local_sum_sq = np.sum(all_eta**2)
    global_sum_sq = domain.comm.allreduce(local_sum_sq, op=MPI.SUM)
    eta = np.sqrt(global_sum_sq)
    
    TRE, ERE, EI = compute_error_metrics(uh, u_numpy, eta, degree_raise=3)
    
    # Test zz estimator
    G = compute_G_tilde(uh)
    ERE_ZZ, EI_ZZ = compute_error_metrics_ZZ(uh, u_numpy, G, degree_raise=3)
    TREs.append(TRE)
    EREs.append(ERE)
    EIs.append(EI)
    ERE_ZZs.append(ERE_ZZ)
    EI_ZZs.append(EI_ZZ)
    hs[i] = 1.0 / N

TREs    = np.array(TREs)
EREs    = np.array(EREs)
EIs     = np.array(EIs)
ERE_ZZs = np.array(ERE_ZZs)
EI_ZZs  = np.array(EI_ZZs)

if domain.comm.rank == 0:
    print(f"{'h':>10} {'TRE':>10} {'p_TRE':>7} "
          f"{'ERE':>10} {'p_ERE':>7} {'EI':>8} "
          f"{'ERE_ZZ':>10} {'p_ZZ':>7} {'EI_ZZ':>8}")
    print("-" * 85)

    for i in range(len(hs)):
        if i == 0:
            # no rate available for the first entry
            p_TRE, p_ERE, p_ZZ = float('nan'), float('nan'), float('nan')
        else:
            log_h  = np.log(hs[i-1]      / hs[i])
            p_TRE  = np.log(TREs[i-1]    / TREs[i])    / log_h
            p_ERE  = np.log(EREs[i-1]    / EREs[i])    / log_h
            p_ZZ   = np.log(ERE_ZZs[i-1] / ERE_ZZs[i]) / log_h

        print(f"{hs[i]:>10.2e} "
              f"{TREs[i]:>10.2e} {p_TRE:>7.2f} "
              f"{EREs[i]:>10.2e} {p_ERE:>7.2f} "
              f"{EIs[i]:>8.4f} "
              f"{ERE_ZZs[i]:>10.2e} {p_ZZ:>7.2f} "
              f"{EI_ZZs[i]:>8.4f}")
    #print(f"h: {hs[i]:.2e}  TRE: {TRE:.2e}  ERE: {ERE:.2e}  EI: {EI:.6f}   ERE_ZZ: {ERE_ZZ:.2e}     EI_ZZ: {EI_ZZ:.3f}")
