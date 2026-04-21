#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Parameters — edit these before running
# ---------------------------------------------------------------------------
PROBLEM="1d"          # "1d", "sphere", or "plan"
K=2                   # 1 = ZZ estimator (P1), 2 = Naga-Zhang estimator (P2)
RESULTS="nz_results_1d" # output directory name (created inside poisson_cube/)


N_LOOP=30             # adaptive iterations per tolerance
TOL_START=1        # first tolerance value
N_TOL=4               # number of tolerance halvings (sequence: TOL_START / 2^i)

HMAX=1.0
HMIN=1e-10
HGRAD=-1
ALPHA=0.25
CORRECTION_FACTOR=1.5

MMG3D="/usr/local/bin/mmg3d_O3"
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/adaptive_poisson_cube.py" \
    --problem           "$PROBLEM"           \
    --k                 "$K"                 \
    --results           "$RESULTS"           \
    --n-loop            "$N_LOOP"            \
    --tol-start         "$TOL_START"         \
    --n-tol             "$N_TOL"             \
    --hmax              "$HMAX"              \
    --hmin              "$HMIN"              \
    --hgrad             "$HGRAD"             \
    --alpha             "$ALPHA"             \
    --correction-factor "$CORRECTION_FACTOR" \
    --mmg3d             "$MMG3D"
