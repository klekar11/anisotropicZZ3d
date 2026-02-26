import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, mesh
from dolfinx.fem import Function, functionspace, form
from dolfinx.fem.petsc import assemble_vector
from petsc4py import PETSc

def compute_error_metrics(domain, V, uh, u_ex, eta):
    
    e = Function(V)
    e.x.array[:] = u_ex.x.array[:] - uh.x.array[:]

    norm_grad_e = np.sqrt(
        fem.assemble_scalar(form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx))
    )
    norm_grad_u = np.sqrt(
        fem.assemble_scalar(form(ufl.inner(ufl.grad(u_ex), ufl.grad(u_ex)) * ufl.dx))
    )
    norm_grad_uh = np.sqrt(
        fem.assemble_scalar(form(ufl.inner(ufl.grad(uh), ufl.grad(uh)) * ufl.dx))
    )

    TRE = norm_grad_e  / norm_grad_u
    ERE = eta          / norm_grad_uh
    EI  = eta          / norm_grad_e

    return TRE, ERE, EI
