#!/usr/bin/env python3
"""Concatenate per-tolerance snap statistics CSVs into one file per face.

Usage:
    python concat_snap_stats.py <results_dir>

Reads snap_stats_<face>.csv from each tol_* subdirectory (processed in
decreasing tolerance order, i.e. the run order), renumbers mesh_idx
sequentially across all tolerances (tol=1 → 0..14, tol=0.5 → 15..29, ...),
writes one combined snap_stats_<face>.csv to <results_dir>, then deletes
the per-tolerance source files.
"""
import argparse
import csv
from pathlib import Path

FACES = ("inner", "outer", "bottom", "top")


def _tol_value(d: Path) -> float:
    return float(d.name.removeprefix("tol_"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate per-tolerance snap statistics CSVs."
    )
    parser.add_argument(
        "results_dir",
        help="Parent directory containing tol_* subdirectories",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise SystemExit(f"Not a directory: {results_dir}")

    # Sort tol_* dirs by tol value descending — that is the run order.
    tol_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("tol_")],
        key=_tol_value,
        reverse=True,
    )
    if not tol_dirs:
        raise SystemExit("No tol_* subdirectories found.")

    print(f"Processing {len(tol_dirs)} tolerance directories (run order):")
    for d in tol_dirs:
        print(f"  {d.name}  (tol={_tol_value(d)})")

    for face in FACES:
        all_rows: list[dict] = []
        to_delete: list[Path] = []
        global_idx = 0

        for tol_dir in tol_dirs:
            csv_path = tol_dir / f"snap_stats_{face}.csv"
            if not csv_path.exists():
                print(f"[skip] {csv_path.relative_to(results_dir)} not found")
                continue
            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                row["mesh_idx"] = global_idx
                all_rows.append(row)
                global_idx += 1
            to_delete.append(csv_path)

        if not all_rows:
            print(f"[face={face}] no data found, skipping")
            continue

        out_path = results_dir / f"snap_stats_{face}.csv"
        fieldnames = list(all_rows[0].keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[face={face}] written → {out_path}  ({len(all_rows)} rows)")

        for p in to_delete:
            p.unlink()
            print(f"  deleted  {p.relative_to(results_dir)}")


if __name__ == "__main__":
    main()
