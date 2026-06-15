#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Parameters — edit these before running
# ---------------------------------------------------------------------------
RESULTS_DIR="cube/nz1d0.01"
ALPHA=0.25          # equidistribution half-band; must match the run's --alpha
# Output image names (saved inside RESULTS_DIR/)
OUT1="tre_vs_nvertices.png"
OUT2="vertices_vs_iteration.png"
OUT3="adapt_split_vs_iteration.png"
OUT4="adapt_split_cells_vs_iteration.png"
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="$SCRIPT_DIR/$RESULTS_DIR"
CSV="$RESULTS/convergence.csv"

python "$SCRIPT_DIR/plot_tre_vs_vertices.py" \
    --csv  "$CSV"                    \
    --out  "$RESULTS/$OUT1"          \
    --out2 "$RESULTS/$OUT2"          \
    --out3 "$RESULTS/$OUT3"          \
    --out4 "$RESULTS/$OUT4"

# Per-tolerance error plots (one PNG per tol_<TOL>/ directory)
python "$SCRIPT_DIR/plot_errors_per_tol.py" \
    --csv     "$CSV"       \
    --alpha   "$ALPHA"     \
    --out-dir "$RESULTS"
