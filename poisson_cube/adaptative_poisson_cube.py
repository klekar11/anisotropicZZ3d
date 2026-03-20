# pyright: reportMissingImports=false, reportMissingModuleSource=false
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Add parent directory to sys.path so sibling modules are importable
# ---------------------------------------------------------------------------
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from adaptive_algo import run_adaptive_poisson  # noqa: E402

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_LOOP           = 5
HMAX             = 1.0
HMIN             = 1e-10
HGRAD            = -1
TOL              = 0.5
ALPHA            = 0.25
CORRECTION_FACTOR = 1.5
MMG3D_EXE        = "/usr/local/bin/mmg3d_O3"

results_dir = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
run_adaptive_poisson(
    results_dir=results_dir,
    n_loop=N_LOOP,
    hmax=HMAX,
    hmin=HMIN,
    hgrad=HGRAD,
    tol=TOL,
    alpha=ALPHA,
    correction_factor=CORRECTION_FACTOR,
    mmg3d_exe=MMG3D_EXE,
)
