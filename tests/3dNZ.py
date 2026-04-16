### Verify 3D post-processing ###
# 3D analogue of the 2D verification script.

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

# Smooth exact solution with nontrivial gradient in every direction.
u = lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.sin(np.pi * x[2])

for pp in post_processing:


    k = 2 if pp == 'NZ_P2' else 1
    Gh = Gh_NZ_P2 if pp == 'NZ_P2' else None
    assert Gh is not None, f"No 3D recovery operator registered for '{pp}'"

    te = np.zeros(Nmesh)   # true error
    ee = np.zeros(Nmesh)   # estimated error
    be = np.zeros(Nmesh)   # better error
    h  = np.zeros(Nmesh)   # mesh size (cube-diagonal convention)

    for i in range(Nmesh):

        N = 2**(i + 1) + 1                     
        h[i] = np.sqrt(3.0) / (N - 1)
        mesh = dolfinx.mesh.create_unit_cube(
            MPI.COMM_WORLD, N, N, N,
            cell_type=dolfinx.mesh.CellType.tetrahedron,
        )

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

        print(f"  [level {i}]  N={N:>3}  #dofs(V)={V.dofmap.index_map.size_global}")

    # Convert from squared H1-seminorm errors to the actual seminorm errors.
    te = np.sqrt(te)
    ee = np.sqrt(ee)
    be = np.sqrt(be)
    ei = ee / te

    print(f"\n{pp}")
    print(f"{'h':>12} {'te':>15} {'ee':>15} {'be':>15} {'ei':>10}")
    print('-' * 72)
    for hi, tei, eei, bei, eii in zip(h, te, ee, be, ei):
        print(f"{hi:12.6e} {tei:15.6e} {eei:15.6e} {bei:15.6e} {eii:10.4f}")

    # Observed convergence rates on the H1-seminorm errors:
    # slope of log10(err) vs log10(h). Expected: te ~ k, be ~ k+1.
    print("\nObserved rates")
    for name, arr in [('te', te), ('ee', ee), ('be', be)]:
        rates = np.log(arr[1:] / arr[:-1]) / np.log(h[1:] / h[:-1])
        print(f"  {name}: " + "  ".join(f"{s:6.2f}" for s in rates))
