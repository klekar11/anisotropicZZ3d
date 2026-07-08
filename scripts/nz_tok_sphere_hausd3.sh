#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Parameters — edit these before running
# ---------------------------------------------------------------------------
PROBLEM="tok-sphere"            # "1d" | "sphere" | "plan" | "tok-sphere" | "tok-wall"
K=2                     # 1 = ZZ estimator (P1), 2 = Naga-Zhang estimator (P2)
RESULTS="../poisson_cube/tokamak/nz_sphere_hausd3" # output directory name (created inside poisson_cube/)

# Starting mesh:
#   leave empty to generate the initial mesh from scratch (cube problems)
#   for tokamak experiments set to the TCV mesh relative to this script:
#     MESH="../tokamak/TCV.mesh"
MESH="../tokamak/TCV.mesh"

N_LOOP=20            # adaptive iterations per tolerance
TOL_START=1          # first tolerance value
N_TOL=4 # number of tolerance halvings (sequence: TOL_START / 2^i)

# mmg3d parameters
HMAX=200
HMIN=1e-7
HGRAD=-1
# adaptive algo params
ALPHA=0.25
CORRECTION_FACTOR=1.5

MMG3D="/usr/local/bin/mmg3d_O3"

# Examples:
#   EXTRA_ARGS=()                                          # no extra flags (cube problems)
#   EXTRA_ARGS=(--nosurf --mmg-extra="-hgradreq -1")       # preserve TCV surface
#   EXTRA_ARGS=(--mmg-extra="-hausd 6.0")                  # hausdorff control, no nosurf
EXTRA_ARGS=(--mmg-extra="-hausd 3.0")
# bissection algo enabled if SNAP_WALLS is true, otherwise disabled
SNAP_WALLS=False
SNAP_R_INNER=200.0
SNAP_R_OUTER=800.0
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MESH_ARG=()
if [ -n "$MESH" ]; then
    MESH_ARG=(--mesh "$MESH")
fi

SNAP_ARGS=()
if [ "$SNAP_WALLS" = "true" ]; then
    SNAP_ARGS=(
        --snap-walls
        --snap-r-inner "$SNAP_R_INNER"
        --snap-r-outer "$SNAP_R_OUTER"
    )
fi

LOG_DIR="$SCRIPT_DIR/$RESULTS"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

python -u "$SCRIPT_DIR/../poisson_cube/adaptive_poisson_cube.py" \
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
    "${SNAP_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
