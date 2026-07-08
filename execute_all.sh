#!/usr/bin/env bash
#
# execute_all.sh — make every script in scripts/ executable and run them.
#
set -euo pipefail

# Directory holding this script, so it works regardless of the caller's cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$SCRIPT_DIR/scripts"

if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Error: directory not found: $TARGET_DIR" >&2
    exit 1
fi

shopt -s nullglob
scripts=("$TARGET_DIR"/*.sh)
shopt -u nullglob

if (( ${#scripts[@]} == 0 )); then
    echo "No .sh scripts found in $TARGET_DIR" >&2
    exit 0
fi

for script in "${scripts[@]}"; do
    echo "==> chmod +x $script"
    chmod +x "$script"
    echo "==> running $script"
    "$script"
    echo "==> done: $script"
done
