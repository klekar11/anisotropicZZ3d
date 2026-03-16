from typing import Tuple, Callable

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import Function, functionspace, form

def compute_error_norms(uh: fem.Function, u_numpy: Callable[[np.ndarray], np.ndarray], degree_raise: int = 3) -> Tuple[float, float, float]:

    domain = uh.function_space.mesh
    family = uh.function_space.ufl_element().family_name
    degree = uh.function_space.ufl_element().degree

    W = functionspace(domain, (family, degree + degree_raise))
    uh_W = Function(W)
    uh_W.interpolate(uh)
    u_ex_W = Function(W)
    u_ex_W.interpolate(u_numpy)
    e_W = Function(W)
    e_W.x.array[:] = u_ex_W.x.array[:] - uh_W.x.array[:]

    norm_grad_e = np.sqrt(fem.assemble_scalar(form(ufl.inner(ufl.grad(e_W), ufl.grad(e_W)) * ufl.dx)))
    norm_grad_u = np.sqrt(fem.assemble_scalar(form(ufl.inner(ufl.grad(u_ex_W), ufl.grad(u_ex_W)) * ufl.dx)))
    norm_grad_uh = np.sqrt(fem.assemble_scalar(form(ufl.inner(ufl.grad(uh_W), ufl.grad(uh_W)) * ufl.dx)))

    return norm_grad_e, norm_grad_u, norm_grad_uh

def compute_error_metrics(uh, u_numpy, eta, degree_raise= 3) -> Tuple[float, float, float]:
    
    norm_grad_e, norm_grad_u, norm_grad_uh = compute_error_norms(uh, u_numpy, degree_raise) 
    
    TRE = norm_grad_e / norm_grad_u
    ERE = eta / norm_grad_uh
    EI  = eta / norm_grad_e
    
    return TRE, ERE, EI

def compute_error_metrics_ZZ(
        uh: fem.Function,
        u_numpy: Callable[[np.ndarray], np.ndarray],
        G: dict[tuple[int, int], np.ndarray],
        degree_raise: int = 3
) -> tuple[float, float]:

    domain = uh.function_space.mesh
    gdim = domain.geometry.dim

    norm_grad_e, _, norm_grad_uh = compute_error_norms(uh, u_numpy, degree_raise)
    eta_ZZ_glob = np.sqrt(sum(np.sum(G[(i, i)]) for i in range(gdim)))
    
    ERE_ZZ = eta_ZZ_glob / norm_grad_uh
    EI_ZZ = eta_ZZ_glob / norm_grad_e
    
    return ERE_ZZ, EI_ZZ
