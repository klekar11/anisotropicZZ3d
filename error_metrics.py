import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, mesh
from dolfinx.fem import Function, functionspace, form
from dolfinx.fem.petsc import assemble_vector
from petsc4py import PETSc

def compute_error_metrics(uh, u_numpy, eta, degree_raise=3):
    domain   = uh.function_space.mesh
    family   = uh.function_space.ufl_element().family_name
    degree   = uh.function_space.ufl_element().degree

    W = functionspace(domain, (family, degree + degree_raise))

    uh_W = Function(W)
    uh_W.interpolate(uh)

    u_ex_W = Function(W)
    u_ex_W.interpolate(u_numpy)

    e_W = Function(W)
    e_W.x.array[:] = u_ex_W.x.array[:] - uh_W.x.array[:]

    norm_grad_e = np.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(form(ufl.inner(ufl.grad(e_W), ufl.grad(e_W)) * ufl.dx)),
        op=MPI.SUM
    ))
    norm_grad_u = np.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(form(ufl.inner(ufl.grad(u_ex_W), ufl.grad(u_ex_W)) * ufl.dx)),
        op=MPI.SUM
    ))
    norm_grad_uh = np.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(form(ufl.inner(ufl.grad(uh_W), ufl.grad(uh_W)) * ufl.dx)),
        op=MPI.SUM
    ))

    TRE = norm_grad_e  / norm_grad_u
    ERE = eta          / norm_grad_uh
    EI  = eta          / norm_grad_e

    return TRE, ERE, EI
