### Verify 3D NZ-P2 with isotropic meshes ###

import numpy as np
import importlib.util
from pathlib import Path
import sys

from mpi4py import MPI
import dolfinx
from dolfinx import fem
import ufl

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eta_estimator1 import compute_G_tilde_nz, compute_anisotropic_eta, compute_jacobian_svd

_NZ_MODULE_PATH = _REPO_ROOT / 'nz_eta_estimatorP2.py'
_NZ_SPEC = importlib.util.spec_from_file_location('nz_eta_estimatorP2', _NZ_MODULE_PATH)
if _NZ_SPEC is None or _NZ_SPEC.loader is None:
    raise ImportError(f'Unable to load {_NZ_MODULE_PATH}')
_NZ_MODULE = importlib.util.module_from_spec(_NZ_SPEC)
_NZ_SPEC.loader.exec_module(_NZ_MODULE)
Gh_NZ_P2 = _NZ_MODULE.Gh

r = 1                # degree raise for the "truth" projection
Nmesh = 5            # number of refinement levels (3D scales cubically!)
post_processing = ('NZ_P2',)   # extend if you port NZ_P1 / ZZ to 3D

u = lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.sin(np.pi * x[2])

def f_smooth_ufl(msh):
    x = ufl.SpatialCoordinate(msh)
    return 3.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[2])

for pp in post_processing:


    k = 2 if pp == 'NZ_P2' else 1
    Gh = Gh_NZ_P2 if pp == 'NZ_P2' else None
    assert Gh is not None, f"No 3D recovery operator registered for '{pp}'"

    te          = np.zeros(Nmesh)
    ee          = np.zeros(Nmesh)
    be          = np.zeros(Nmesh)
    eta_ZZ_arr  = np.zeros(Nmesh)
    eta_a_arr   = np.zeros(Nmesh)
    ar_arr      = np.zeros(Nmesh)
    h           = np.zeros(Nmesh)

    for i in range(Nmesh):

        N = 2**(i + 1) + 1
        h[i] = np.sqrt(3.0) / (N - 1)
        mesh = dolfinx.mesh.create_unit_cube(
            MPI.COMM_WORLD, N, N, N,
            cell_type=dolfinx.mesh.CellType.tetrahedron,
        )

        f_ufl = f_smooth_ufl(mesh)

        # FE space of degree k, and the recovered gradient
        V = fem.functionspace(mesh, ("CG", k))
        uh = fem.Function(V)
        uh.interpolate(u)
        Ghuh = Gh(uh)

        # Richer space of degree k+r for the "exact" gradient proxy
        Vr = fem.functionspace(mesh, ("CG", k + r))
        ur = fem.Function(Vr)
        ur.interpolate(u)

        te[i] = fem.assemble_scalar(fem.form(
            ufl.dot(ufl.grad(uh - ur), ufl.grad(uh - ur)) * ufl.dx))
        ee[i] = fem.assemble_scalar(fem.form(
            ufl.dot(ufl.grad(uh) - Ghuh, ufl.grad(uh) - Ghuh) * ufl.dx))
        be[i] = fem.assemble_scalar(fem.form(
            ufl.dot(ufl.grad(ur) - Ghuh, ufl.grad(ur) - Ghuh) * ufl.dx))

        # EI_ZZ and EI_A (NZ-based, cf. 3dNZ_aniso.py)
        G_nz, _ = compute_G_tilde_nz(uh)
        gdim = mesh.geometry.dim
        eta_ZZ_arr[i] = np.sqrt(max(sum(np.sum(G_nz[(j, j)]) for j in range(gdim)), 0.0))

        eta_a_cells, _, _ = compute_anisotropic_eta(uh, f_ufl, k=k)
        eta_a_arr[i] = np.sqrt(np.sum(eta_a_cells))

        svd = compute_jacobian_svd(mesh)
        ar_arr[i] = np.mean(svd["AR"])

        print(f"  [level {i}]  N={N:>3}  #dofs(V)={V.dofmap.index_map.size_global}"
              f"  mean_AR={ar_arr[i]:.3f}")

    te     = np.sqrt(te)
    ee     = np.sqrt(ee)
    be     = np.sqrt(be)
    ERE_Gh = ee / te
    EI_ZZ  = eta_ZZ_arr / te
    EI_A   = eta_a_arr  / te

    print(f"\n{pp}")
    print(f"{'h':>12} {'te':>15} {'ee':>15} {'be':>15}"
          f" {'ERE_Gh':>8} {'EI_ZZ':>8} {'EI_A':>8} {'mean_AR':>9}")
    print('-' * 100)
    for hi, tei, eei, bei, ere, eizz, eia, ari in zip(
            h, te, ee, be, ERE_Gh, EI_ZZ, EI_A, ar_arr):
        print(f"{hi:12.6e} {tei:15.6e} {eei:15.6e} {bei:15.6e}"
              f" {ere:8.4f} {eizz:8.4f} {eia:8.4f} {ari:9.4f}")

    # Observed convergence rates on the H1-seminorm errors.
    print("\nObserved rates")
    for name, arr in [('te', te), ('ee', ee), ('be', be)]:
        rates = np.log(arr[1:] / arr[:-1]) / np.log(h[1:] / h[:-1])
        print(f"  {name}: " + "  ".join(f"{s:6.2f}" for s in rates))
