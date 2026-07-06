#!/usr/bin/env python3
"""Plot snap statistics produced by the adaptive tokamak snapping algorithm.

Usage:
    python plot_snap_stats.py <results_dir>

Looks for snap_stats_inner.csv, snap_stats_outer.csv, snap_stats_bottom.csv,
and snap_stats_top.csv inside <results_dir>.  One PNG per face is saved there.
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FACES = ("inner", "outer", "bottom", "top")


def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def plot_face(rows: list[dict], face: str, out_path: Path) -> None:
    mesh_idx = [r["mesh_idx"] for r in rows]
    total    = [r["total"]    for r in rows]
    snapped  = [r["snapped"]  for r in rows]
    bisected = [r["bisected"] for r in rows]
    mean_d   = [r["mean_dist_bisected"] for r in rows]
    std_d    = [r["std_dist_bisected"]  for r in rows]
    max_d    = [r["max_dist_bisected"]  for r in rows]

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()

    l1, = ax1.plot(mesh_idx, total,    "b-o",  ms=5, label="total vertices")
    l2, = ax1.plot(mesh_idx, snapped,  "g--s", ms=5, label="snapped")
    l3, = ax1.plot(mesh_idx, bisected, "r:^",  ms=5, label="not exact (fallback to bisection)")

    l4, = ax2.plot(mesh_idx, mean_d, color="orchid",     linestyle="-",  marker="o", ms=4,
                   alpha=0.85, label="mean dist")
    l5, = ax2.plot(mesh_idx, std_d,  color="goldenrod",  linestyle="--", marker="s", ms=4,
                   alpha=0.85, label="std dist")
    l6, = ax2.plot(mesh_idx, max_d,  color="slategray",  linestyle=":",  marker="^", ms=4,
                   alpha=0.85, label="max dist")

    ax1.set_xlabel("Mesh index")
    ax1.set_ylabel("Vertex count")
    ax2.set_ylabel("Residual distance to exact surface")

    lines  = [l1, l2, l3, l4, l5, l6]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot snap statistics from adaptive tokamak snapping CSVs."
    )
    parser.add_argument("results_dir", help="Directory containing snap_stats_*.csv files")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise SystemExit(f"Not a directory: {results_dir}")

    found = False
    for face in FACES:
        csv_path = results_dir / f"snap_stats_{face}.csv"
        if not csv_path.exists():
            print(f"[skip] {csv_path.name} not found")
            continue
        rows = load_csv(csv_path)
        if not rows:
            print(f"[skip] {csv_path.name} is empty")
            continue
        found = True
        out_path = results_dir / f"snap_stats_{face}.png"
        plot_face(rows, face, out_path)

    if not found:
        raise SystemExit("No snap_stats_*.csv files found — nothing to plot.")


if __name__ == "__main__":
    main()
