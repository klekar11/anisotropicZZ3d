import numpy as np
import ufl
from ufl import SpatialCoordinate, TestFunction, TrialFunction, div, grad, inner, dx
from mpi4py import MPI
from dolfinx import default_scalar_type, fem
from dolfinx.fem import functionspace, Function, form, dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_square, locate_entities_boundary, create_unit_cube

from eta_estimator import (compute_iso_eta, compute_G_tilde,
                       compute_anisotropic_eta)
from error_metrics import compute_error_norms


def u_ex(mod):
    return lambda x: mod.sin(mod.pi * x[0]) * mod.sin(2 * mod.pi * x[1]) * mod.sin(mod.pi * x[2])

u_numpy = u_ex(np)
u_ufl   = u_ex(ufl)


def solve_poisson_anisotropic(N1: int, N2: int, N3: int, degree: int = 1):
    """
    N1 : number of cells in x1 direction
    N2 : number of cells in x2 direction
    Defining N1 != N2 creates an anisotropic mesh.
    """
    msh = create_unit_cube(MPI.COMM_SELF, N1, N2, N3)
    x   = SpatialCoordinate(msh)
    f   = -div(grad(u_ufl(x)))
    V   = functionspace(msh, ("Lagrange", degree))
    u   = TrialFunction(V)
    v   = TestFunction(V)
    a   = inner(grad(u), grad(v)) * dx
    L   = f * v * dx
    facets = locate_entities_boundary(
        msh, msh.topology.dim - 1,
        lambda x: np.full(x.shape[1], True)
    )
    dofs = locate_dofs_topological(V, msh.topology.dim - 1, facets)
    bcs  = [dirichletbc(default_scalar_type(0), dofs, V)]
    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "gmres", "pc_type": "none"},
        petsc_options_prefix="poisson_aniso_",
    )
    return problem.solve(), f

# ------------------------------------------------------------------
# Anisotropic mesh refinement study
# N1 fixed fine in x1, N2 coarsening ratio controlled separately
# h1 = 1/N1, h2 = 1/N2 — reported separately to show anisotropy
# ------------------------------------------------------------------
configs = [
    # (1,  4, 4),
    (2,  8, 8),
    (4, 16, 16),
    (8, 32, 32),
    (16, 64, 64),
    #(128, 128, 128)
]

EI_isos = []
EI_ZZs  = []
EI_as   = []
h1s     = []
h2s     = []
h3s     = []

for N1, N2, N3 in configs:
    uh, f = solve_poisson_anisotropic(N1, N2, N3, degree=1)
    domain = uh.function_space.mesh

    h1 = 1.0 / N1
    h2 = 1.0 / N2
    h3 = 1.0 / N3
    h1s.append(h1)
    h2s.append(h2)
    h3s.append(h3)

    # --- true error norm ---
    norm_grad_e, _, norm_grad_uh = compute_error_norms(uh, u_numpy, degree_raise=3)

    # --- isotropic residual estimator ---
    eta_iso_cells = compute_iso_eta(uh, f)
    eta_iso = np.sqrt(np.sum(eta_iso_cells**2))
    EI_iso = eta_iso / norm_grad_e
    EI_isos.append(EI_iso)

    # --- ZZ estimator ---
    G = compute_G_tilde(uh)
    gdim = domain.geometry.dim
    eta_ZZ = np.sqrt(max(sum(np.sum(G[(i, i)]) for i in range(gdim)), 0.0))
    EI_ZZ = eta_ZZ / norm_grad_e
    EI_ZZs.append(EI_ZZ)

    # --- anisotropic estimator ---
    eta_a_cells = compute_anisotropic_eta(uh, f)
    eta_a = np.sqrt(np.sum(eta_a_cells))
    EI_a = eta_a / norm_grad_e
    EI_as.append(EI_a)

    # print(f"N1={N1:>3} N2={N2:>3}  h1={h1:.3e}  h2={h2:.3e}"
    #       f"||∇e||={norm_grad_e:.3e}"
    #       f"EI_iso={EI_iso:.4f}  EI_ZZ={EI_ZZ:.4f}  EI_a={EI_a:.4f}")

print(f"\n{'h1':>10} {'h2':>10} {'h3':>10} {'EI_iso':>10} {'EI_ZZ':>10} {'EI_a':>10}")
print("-" * 65)
for i in range(len(configs)):
    print(f"{h1s[i]:>10.3e} {h2s[i]:>10.3e} {h3s[i]:>10.3e}"
          f"{EI_isos[i]:>10.4f} {EI_ZZs[i]:>10.4f} {EI_as[i]:>10.4f}")