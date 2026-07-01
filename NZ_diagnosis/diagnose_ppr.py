# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Per-iteration PPR / Naga-Zhang diagnostic.

Lightweight version of PPRConvergenceTracker.record — same A–F quantities
printed at every adaptive loop iteration for live debugging.

  A = ||∇u_h − ∇u||             true FE gradient error
  B = ||∇u_h − G_h(u_h)||       PPR–FE residual
  C = ||G_h(u_h) − ∇u||         PPR recovery error vs. exact
  D = ||∇u − G_h(I²u)||         recovery error on the P2 interpolant
  E = ||G_h(u_h) − G_h(I²u)||   PPR sensitivity (PDE vs. interpolation error)
  F = ||∇(I²u) − ∇u_h||         supercloseness / FE vs. interpolant

All norms are global L² over the whole domain.

Expected asymptotic rates (P2, h → 0):
  A ~ O(h²),  B ~ O(h²),  C ~ O(h³)  (superconvergent PPR)
  D ~ O(h³),  E ~ o(h²),  F ~ O(h³)  (supercloseness)
"""

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import form


def diagnose_ppr(
    u_h,
    u_numpy,
    Gh_func,
    degree_raise: int = 4,
) -> None:
    """Print PPR diagnostic quantities A–F for one adaptive iteration.

    Parameters
    ----------
    u_h          : Converged FEM solution, a P2 ``fem.Function``.
    u_numpy      : Exact solution numpy callback, signature ``u_numpy(x) -> array``.
    Gh_func      : PPR recovery callable — ``nz_eta_estimatorP2.Gh``.
                   Signature: ``Gh_func(u_h) -> fem.Function`` (P2 vector).
    degree_raise : Polynomial degree of the high-order reference space used to
                   approximate ∇u from u_numpy.  Default 4 gives a CG-6 reference
                   for a P2 solution, which is well above the O(h³) target accuracy.
    """
    V    = u_h.function_space
    mesh = V.mesh
    deg  = V.ufl_element().degree

    # I²u — exact solution projected onto the same P2 space as u_h
    I2u = fem.Function(V)
    I2u.interpolate(u_numpy)

    # Reference exact gradient: lift u_numpy to a high-degree CG space and
    # take its UFL gradient.  Evaluated at Gauss points during assembly, so
    # it gives a spectral-quality approximation of ∇u without needing a
    # separate gradient callback.
    W      = fem.functionspace(mesh, ("CG", deg + degree_raise))
    u_ex_W = fem.Function(W)
    u_ex_W.interpolate(u_numpy)
    grad_u_ex = ufl.grad(u_ex_W)   # UFL expression, not pre-interpolated

    # PPR recovered gradients (both live in the P2 vector space)
    G_nz   = Gh_func(u_h)   # G_h(u_h):  recovery of FE solution
    Gh_I2u = Gh_func(I2u)   # G_h(I²u): recovery of P2 interpolant

    def _l2(expr):
        """Global L² norm of any scalar-or-vector UFL expression."""
        return float(np.sqrt(fem.assemble_scalar(form(ufl.inner(expr, expr) * ufl.dx))))

    # ---- A, B, C --------------------------------------------------------
    A = _l2(ufl.grad(u_h) - grad_u_ex)    # ||∇u_h − ∇u||
    B = _l2(ufl.grad(u_h) - G_nz)         # ||∇u_h − G_h(u_h)||
    C = _l2(G_nz - grad_u_ex)             # ||G_h(u_h) − ∇u||

    # ---- D, E, F (interpolant decomposition) ----------------------------
    D = _l2(grad_u_ex - Gh_I2u)                     # ||∇u − G_h(I²u)||
    E = _l2(G_nz - Gh_I2u)                          # ||G_h(u_h) − G_h(I²u)||
    F = _l2(ufl.grad(I2u) - ufl.grad(u_h))         # ||∇(I²u) − ∇u_h||

    norm_grad_uh = _l2(ufl.grad(u_h))
    rel = lambda v: v / norm_grad_uh if norm_grad_uh > 1e-30 else float("nan")

    print(f"  A = ||∇u_h − ∇u||             = {A:.4e}  (rel: {rel(A):.4e})")
    print(f"  B = ||∇u_h − G_h(u_h)||       = {B:.4e}  (rel: {rel(B):.4e})")
    print(f"  C = ||G_h(u_h) − ∇u||         = {C:.4e}  (rel: {rel(C):.4e})")
    print(f"  D = ||∇u − G_h(I²u)||         = {D:.4e}  (rel: {rel(D):.4e})")
    print(f"  E = ||G_h(u_h) − G_h(I²u)||   = {E:.4e}  (rel: {rel(E):.4e})")
    print(f"  F = ||∇(I²u) − ∇u_h||         = {F:.4e}  (rel: {rel(F):.4e})")
    print(f"  Checks: A≤B+C={B+C:.3e},  C≤D+E={D+E:.3e},  B~O(h²) C~O(h³)")
