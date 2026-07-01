# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Per-cell PPR / Naga-Zhang diagnostic written to XDMF.

Writes DG-0 (piecewise-constant) fields for the cell-wise L² contributions
to the six PPR error quantities A–F so they can be inspected in ParaView.

  A_cell = sqrt( |∇u_h − ∇u|²  ) at cell centroid   × sqrt(|K|)
  ...

Each field stores the approximate per-cell L² contribution
(centroid-rule quadrature × cell volume).  The sum of squares of all cells
equals the global L² norm squared (to 1-point quadrature accuracy).
"""

from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import form
from dolfinx.io import XDMFFile


def write_cellwise_diagnostic(
    u_h,
    u_numpy,
    Gh_func,
    grad_ex_factory,
    output_path: "str | Path" = "ppr_cellwise_diag.xdmf",
) -> None:
    """Compute and write per-cell PPR diagnostic fields to XDMF.

    Parameters
    ----------
    u_h             : Converged FEM solution, a P2 ``fem.Function``.
    u_numpy         : Exact solution numpy callback ``u_numpy(x) -> array``.
    Gh_func         : PPR recovery callable — ``nz_eta_estimatorP2.Gh``.
    grad_ex_factory : Factory ``grad_ex_factory(msh) -> fem.Function`` returning
                      the exact gradient as a high-degree vector fem.Function.
    output_path     : Path for the output ``.xdmf`` file.
    """
    V    = u_h.function_space
    mesh = V.mesh

    # ---- Build all ingredient functions ------------------------------
    I2u = fem.Function(V)
    I2u.interpolate(u_numpy)

    grad_u_ex = grad_ex_factory(mesh)   # high-degree vector fem.Function
    G_nz      = Gh_func(u_h)            # G_h(u_h):  P2 vector
    Gh_I2u    = Gh_func(I2u)            # G_h(I²u): P2 vector

    # ---- DG-0 space — one DOF per cell --------------------------------
    DG0 = fem.functionspace(mesh, ("DG", 0))
    pts = DG0.element.interpolation_points  # cell centroids

    # Cell volumes (needed to turn pointwise values into cell integrals)
    vol_fn = fem.Function(DG0)
    vol_fn.interpolate(fem.Expression(ufl.CellVolume(mesh), pts))

    # ---- Helper: evaluate |expr|² at centroids, multiply by volume ----
    def _cellwise(name: str, expr) -> fem.Function:
        fn = fem.Function(DG0)
        fn.name = name
        fn.interpolate(fem.Expression(ufl.inner(expr, expr), pts))
        # fn holds pointwise |expr|²; multiply by |K| → cell integral approx
        fn.x.array[:] = np.sqrt(np.maximum(fn.x.array * vol_fn.x.array, 0.0))
        return fn

    field_defs = [
        ("A_cell", ufl.grad(u_h) - grad_u_ex),          # ||∇u_h − ∇u||
        ("B_cell", ufl.grad(u_h) - G_nz),               # ||∇u_h − G_h(u_h)||
        ("C_cell", G_nz - grad_u_ex),                   # ||G_h(u_h) − ∇u||
        ("D_cell", grad_u_ex - Gh_I2u),                 # ||∇u − G_h(I²u)||
        ("E_cell", G_nz - Gh_I2u),                      # ||G_h(u_h) − G_h(I²u)||
        ("F_cell", ufl.grad(I2u) - ufl.grad(u_h)),     # ||∇(I²u) − ∇u_h||
    ]

    output_path = Path(output_path)
    with XDMFFile(mesh.comm, str(output_path), "w") as xf:
        xf.write_mesh(mesh)
        for name, expr in field_defs:
            fn = _cellwise(name, expr)
            xf.write_function(fn)

    # Print global norms for a quick sanity check (uses exact assembly)
    print(f"  [ppr_cellwise_diag] Saved → {output_path}")
    for name, expr in field_defs:
        val = float(np.sqrt(fem.assemble_scalar(form(ufl.inner(expr, expr) * ufl.dx))))
        print(f"  [ppr_cellwise_diag]  ||{name[0]}|| = {val:.4e}")
