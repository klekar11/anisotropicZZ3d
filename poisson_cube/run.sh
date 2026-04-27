#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Parameters — edit these before running
# ---------------------------------------------------------------------------
PROBLEM="tok-sphere"            # "1d" | "sphere" | "plan" | "tok-sphere"
K=1                     # 1 = ZZ estimator (P1), 2 = Naga-Zhang estimator (P2)
RESULTS="../tokamak/zz_nosurf/" # output directory name (created inside poisson_cube/)

# Starting mesh:
#   leave empty to generate the initial mesh from scratch (cube problems)
#   for tok-sphere set to the TCV mesh relative to this script:
#     MESH="$SCRIPT_DIR/../tokamak/TCV.mesh"
MESH="../tokamak/TCV.mesh"

N_LOOP=15             # adaptive iterations per tolerance
TOL_START=1          # first tolerance value
N_TOL=3               # number of tolerance halvings (sequence: TOL_START / 2^i)

HMAX=200
HMIN=1e-5
HGRAD=-1
ALPHA=0.25
CORRECTION_FACTOR=1.5

MMG3D="/usr/local/bin/mmg3d_O3"

# Extra flags forwarded to the Python script.
#
# --nosurf                    preserve the TCV surface mesh during MMG3D adaptation
# --mmg-extra="..."           additional MMG3D flags as a single quoted string
#
# IMPORTANT: always use = to attach the value to --mmg-extra (no space).
# A space would make bash pass two words and argparse would misread the
# leading "-" of the MMG flag as a new Python argument.
#
# Examples:
#   EXTRA_ARGS=()                                   # no extra flags (cube problems)
#   EXTRA_ARGS=(--nosurf)                           # preserve TCV surface
#   EXTRA_ARGS=(--mmg-extra="-hausd 6.0")           # hausdorff control, no nosurf
#   EXTRA_ARGS=(--nosurf --mmg-extra="-hausd 6.0")  # both
#   EXTRA_ARGS=(--nosurf --mmg-extra="-hausd 6.0 -ar 21")  # multiple MMG flags
EXTRA_ARGS=(--nosurf)
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MESH_ARG=()
if [ -n "$MESH" ]; then
    MESH_ARG=(--mesh "$MESH")
fi

LOG_DIR="$SCRIPT_DIR/$RESULTS"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

python -u "$SCRIPT_DIR/adaptive_poisson_cube.py" \
    --problem           "$PROBLEM"           \
    --k                 "$K"                 \
    --results           "$RESULTS"           \
    "${MESH_ARG[@]}"                         \
    --n-loop            "$N_LOOP"            \
    --tol-start         "$TOL_START"         \
    --n-tol             "$N_TOL"             \
    --hmax              "$HMAX"              \
    --hmin              "$HMIN"              \
    --hgrad             "$HGRAD"             \
    --alpha             "$ALPHA"             \
    --correction-factor "$CORRECTION_FACTOR" \
    --mmg3d             "$MMG3D"             \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
