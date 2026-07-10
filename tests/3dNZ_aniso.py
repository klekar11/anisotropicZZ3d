### Verify 3D NZ-P2 recovery on anisotropic meshes ###
# Mesh configs follow the (N1, N2, N3) format of 3D_Poisson_aniso_error.py.

import numpy as np
import importlib.util
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ufl
from ufl import SpatialCoordinate, TrialFunction, TestFunction, grad, inner, dx as udx
from mpi4py import MPI
import dolfinx
from dolfinx import fem
from dolfinx.fem import (functionspace, Function, form,
                         dirichletbc, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_cube, locate_entities_boundary

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_NZ_MODULE_PATH = _REPO_ROOT / 'nz_eta_estimatorP2.py'
_NZ_SPEC = importlib.util.spec_from_file_location('nz_eta_estimatorP2', _NZ_MODULE_PATH)
if _NZ_SPEC is None or _NZ_SPEC.loader is None:
    raise ImportError(f'Unable to load {_NZ_MODULE_PATH}')
_NZ_MODULE = importlib.util.module_from_spec(_NZ_SPEC)
_NZ_SPEC.loader.exec_module(_NZ_MODULE)
Gh_NZ_P2 = _NZ_MODULE.Gh

from eta_estimator1 import compute_G_tilde_nz, compute_anisotropic_eta

USE_BOUNDARY_LAYER =False # set True for the 1D tanh boundary layer
EPSILON            = 0.1    # layer thickness

# anisotropy ratio = N3 / N1;  N1=N2 fixed, N3 = N1 * ratio
configs = [
    (2, 2, 8),
    (4, 4, 16),
    (8, 8, 32),
	(8,8,48),
    (16, 16, 64), 
    (16, 16, 82), 
#	(16, 16, 92),
#	(16, 16, 102),
#    (16, 16, 112),
#    (16, 16, 128),
#    (16, 16, 142)
]

r = 3   # degree raise for the "truth" interpolation
k = 2   # P2 NZ recovery

# Smooth reference (USE_BOUNDARY_LAYER=False)
def u_smooth(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.sin(np.pi * x[2])

# 1D tanh layer in z — formula from problems.py _problem_1d adapted to x[2]
def u_tanh(x):
    return np.tanh(x[2] / EPSILON)

def f_tanh_ufl(msh):
    x = SpatialCoordinate(msh)
    t = ufl.tanh(x[2] / EPSILON)
    return (2.0 / EPSILON**2) * t * (1.0 - t**2)

def f_smooth_ufl(msh):
    x = SpatialCoordinate(msh)
    return 3.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[2])

u_numpy = u_tanh if USE_BOUNDARY_LAYER else u_smooth

def solve_poisson(msh, degree):
    """Solve -Δu = f with non-homogeneous Dirichlet BCs from u_tanh."""
    V  = functionspace(msh, ("Lagrange", degree))
    u  = TrialFunction(V)
    v  = TestFunction(V)
    a  = inner(grad(u), grad(v)) * udx
    L  = f_tanh_ufl(msh) * v * udx
    facets = locate_entities_boundary(msh, msh.topology.dim - 1,
                                      lambda x: np.full(x.shape[1], True))
    dofs   = locate_dofs_topological(V, msh.topology.dim - 1, facets)
    g_fn   = Function(V)
    g_fn.interpolate(u_tanh)
    bcs    = [dirichletbc(g_fn, dofs)]
    return LinearProblem(a, L, bcs=bcs,
                                                 petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": "1e-12"},
                                                 petsc_options_prefix="poisson_nz_aniso_").solve()

pp = 'NZ_P2'
Gh = Gh_NZ_P2

n           = len(configs)
te          = np.zeros(n)
ee          = np.zeros(n)
be          = np.zeros(n)
de          = np.zeros(n)
Ee          = np.zeros(n)
fe          = np.zeros(n)
eta_ZZ_arr  = np.zeros(n)
eta_a_arr   = np.zeros(n)
h_xy_arr    = np.zeros(n)
h_z_arr     = np.zeros(n)

for i, (N1, N2, N3) in enumerate(configs):
    mesh = create_unit_cube(MPI.COMM_WORLD, N1, N2, N3,
                            cell_type=dolfinx.mesh.CellType.tetrahedron)
    h_xy_arr[i] = 1.0 / N1
    h_z_arr[i]  = 1.0 / N3

    f_ufl = f_tanh_ufl(mesh) if USE_BOUNDARY_LAYER else f_smooth_ufl(mesh)

    if USE_BOUNDARY_LAYER:
        uh = solve_poisson(mesh, k)
    else:
        V  = functionspace(mesh, ("CG", k))
        uh = Function(V)
        uh.interpolate(u_numpy)

    Ghuh = Gh(uh)

    # P(k+r) interpolation of exact solution serves as "truth"
    Vr = functionspace(mesh, ("CG", k + r))
    ur = Function(Vr)
    ur.interpolate(u_numpy)

    te[i] = fem.assemble_scalar(form(
        ufl.dot(ufl.grad(uh - ur), ufl.grad(uh - ur)) * ufl.dx))
    ee[i] = fem.assemble_scalar(form(
        ufl.dot(ufl.grad(uh) - Ghuh, ufl.grad(uh) - Ghuh) * ufl.dx))
    be[i] = fem.assemble_scalar(form(
        ufl.dot(ufl.grad(ur) - Ghuh, ufl.grad(ur) - Ghuh) * ufl.dx))

    # I²u: P2 interpolant of exact solution in same space as uh
    I2u = Function(uh.function_space)
    I2u.interpolate(u_numpy)
    Gh_I2u = Gh(I2u)

    de[i] = fem.assemble_scalar(form(
        ufl.dot(ufl.grad(ur) - Gh_I2u, ufl.grad(ur) - Gh_I2u) * ufl.dx))
    Ee[i] = fem.assemble_scalar(form(
        ufl.dot(Ghuh - Gh_I2u, Ghuh - Gh_I2u) * ufl.dx))
    fe[i] = fem.assemble_scalar(form(
        ufl.dot(ufl.grad(I2u) - ufl.grad(uh), ufl.grad(I2u) - ufl.grad(uh)) * ufl.dx))

    # EI_ZZ: NZ-based EI
    G_nz, _ = compute_G_tilde_nz(uh)
    gdim = mesh.geometry.dim
    eta_ZZ_arr[i] = np.sqrt(max(sum(np.sum(G_nz[(j, j)]) for j in range(gdim)), 0.0))

    # EI_A: NZ-based EI^A
    eta_a_cells, _, _ = compute_anisotropic_eta(uh, f_ufl, k=k)
    eta_a_arr[i] = np.sqrt(np.sum(eta_a_cells))

    ratio = N3 // N1
    ndofs = uh.function_space.dofmap.index_map.size_global
    print(f"  [ratio {ratio:>4}x]  ({N1},{N2},{N3})  #dofs={ndofs}")

te    = np.sqrt(te)
ee    = np.sqrt(ee)
be    = np.sqrt(be)
de    = np.sqrt(de)
Ee    = np.sqrt(Ee)
fe    = np.sqrt(fe)
EI_ZZ = eta_ZZ_arr / te   # NZ-ZZ estimator / true error
EI_A  = eta_a_arr  / te   # NZ anisotropic estimator / true error

label = f'1D tanh layer in z (eps={EPSILON})' if USE_BOUNDARY_LAYER else 'smooth sin^3'
print(f"\n{pp}  ({label})")
print(f"{'ratio':>8} {'h_xy':>12} {'h_z':>12} "
      f"{'A(te)':>13} {'B(ee)':>13} {'C(be)':>13} "
      f"{'EI_ZZ':>8} {'EI_A':>8}")
print('-' * 100)
for (N1, _, N3), hxy, hz, Ai, Bi, Ci, eizz, eia in zip(
        configs, h_xy_arr, h_z_arr, te, ee, be, EI_ZZ, EI_A):
    ratio = N3 // N1
    print(f"{ratio:>8} {hxy:12.6e} {hz:12.6e} "
          f"{Ai:13.6e} {Bi:13.6e} {Ci:13.6e} "
          f"{eizz:8.4f} {eia:8.4f}")

print("\nObserved rates (w.r.t. h_z)")
for name, arr in [('A(te)', te), ('B(ee)', ee), ('C(be)', be)]:
    rates = np.log(arr[1:] / arr[:-1]) / np.log(h_z_arr[1:] / h_z_arr[:-1])
    print(f"  {name}: " + "  ".join(f"{s:6.2f}" for s in rates))

_FINFO = {
    "A": {"label": r"$A=\|\nabla u_h-\nabla u\|$",       "color": "#1f77b4", "marker": "o", "ls": "-"},
    "B": {"label": r"$B=\|\nabla u_h-G_h u_h\|$",        "color": "#ff7f0e", "marker": "s", "ls": "-"},
    "C": {"label": r"$C=\|G_h u_h-\nabla u\|$",          "color": "#2ca02c", "marker": "^", "ls": "-"},
    "D": {"label": r"$D=\|\nabla u-G_h(I^2 u)\|$",       "color": "#d62728", "marker": "D", "ls": "--"},
    "E": {"label": r"$E=\|G_h u_h-G_h(I^2 u)\|$",       "color": "#9467bd", "marker": "v", "ls": "--"},
    "F": {"label": r"$F=\|\nabla(I^2 u)-\nabla u_h\|$",  "color": "#8c564b", "marker": "P", "ls": "--"},
}
_arrs  = {"A": te, "B": ee, "C": be, "D": de, "E": Ee, "F": fe}
_yoff  = {"A": 1.4, "B": 0.6, "C": 1.4, "D": 0.6, "E": 1.4, "F": 0.6}


def _save_plot(subset, title, out_stem):
    fig, ax = plt.subplots(figsize=(8, 6))
    h = h_z_arr
    for name in subset:
        vals = _arrs[name]
        info = _FINFO[name]
        if np.all(vals > 0):
            ax.loglog(h, vals, marker=info["marker"], linestyle=info["ls"],
                      color=info["color"], linewidth=2, markersize=7,
                      label=info["label"])
    if n >= 2:
        h_ref = np.array([h.min(), h.max()])
        ax.loglog(h_ref, (te[-1] / h[-1]**2) * h_ref**2,
                  "--", color="gray", lw=1, alpha=0.5, label=r"$O(h^2)$")
        if de[-1] > 0:
            ax.loglog(h_ref, (de[-1] / h[-1]**3) * h_ref**3,
                      ":", color="gray", lw=1, alpha=0.5, label=r"$O(h^3)$")
        for name in subset:
            vals = _arrs[name]
            if not np.all(vals > 0):
                continue
            rate = np.log(vals[-1] / vals[-2]) / np.log(h[-1] / h[-2])
            xmid = np.sqrt(h[-1] * h[-2])
            ymid = np.sqrt(vals[-1] * vals[-2])
            ax.annotate(f"{rate:.2f}", xy=(xmid, ymid * _yoff[name]),
                        fontsize=9, color=_FINFO[name]["color"],
                        fontweight="bold", ha="center")
    ax.set_xlabel(r"$h_z$", fontsize=13)
    ax.set_ylabel(r"$L^2$ gradient error", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    for ext in (".pdf", ".png"):
        p = Path(__file__).resolve().parent / (out_stem + ext)
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {p}")
    plt.close(fig)


_sfx = f"_eps{EPSILON}_{'BL' if USE_BOUNDARY_LAYER else 'smooth'}"
_save_plot(["A", "B", "C"],
           f"PPR convergence: FE errors  (eps={EPSILON})",
           f"3dNZ_aniso_ABC{_sfx}")
_save_plot(["D", "E", "F"],
           f"PPR convergence: interpolant decomposition  (eps={EPSILON})",
           f"3dNZ_aniso_DEF{_sfx}")
_save_plot(list(_FINFO.keys()),
           f"PPR convergence: all errors  (eps={EPSILON})",
           f"3dNZ_aniso_all{_sfx}")
