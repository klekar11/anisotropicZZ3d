#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Parameters — edit these before running
# ---------------------------------------------------------------------------
RESULTS_DIR="../mmg_tests/zz_tok_nosurf"
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
