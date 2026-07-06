#!/usr/bin/env python3
"""
plot_surface_dev.py
===================
For each tol_<value>/ subdirectory found inside a given results directory,
iterate over all meshes (in numerical order) inside the meshes/ subdirectory,
compute the mean + std of vertex deviations from the expected cylindrical
radius at each iteration, and produce a single convergence plot.

The geometry analysis reuses exactly the same logic as check_surface_verts.py.

Usage
-----
    python plot_surface_dev.py RESULTS_DIR OUTPUT_FIG [OPTIONS]

    RESULTS_DIR   directory containing tol_*/meshes/ sub-trees
    OUTPUT_FIG    path for the saved figure (e.g. dev_convergence.png)

Options (same as check_surface_verts.py)
------------------------------------------
--hausd VAL      Hausdorff tolerance used in the MMG run.  Default 5.0.
--r-outer VAL    Expected outer-wall cylindrical radius.  Auto-detected if omitted.
--r-inner VAL    Expected inner-wall cylindrical radius.  Auto-detected if omitted.
--verbose        Print per-iteration statistics to stdout.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Medit parser  (identical to check_surface_verts.py)
# ---------------------------------------------------------------------------

def read_medit(path: Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    vertices: list[list[float]] = []
    tri_by_ref: dict[int, list[list[int]]] = {}

    with open(path) as f:
        lines = f.readlines()

    i = 0
    n = len(lines)

    def skip() -> None:
        nonlocal i
        while i < n and (lines[i].strip() == "" or lines[i].strip().startswith("#")):
            i += 1

    while i < n:
        token = lines[i].strip().split()
        if not token:
            i += 1
            continue
        kw = token[0]

        if kw == "Vertices":
            i += 1; skip()
            count = int(lines[i].strip()); i += 1
            for _ in range(count):
                skip()
                p = lines[i].strip().split()
                vertices.append([float(p[0]), float(p[1]), float(p[2])])
                i += 1
            continue

        if kw == "Triangles":
            i += 1; skip()
            count = int(lines[i].strip()); i += 1
            for _ in range(count):
                skip()
                p = lines[i].strip().split()
                v0, v1, v2 = int(p[0]) - 1, int(p[1]) - 1, int(p[2]) - 1
                ref = int(p[3])
                tri_by_ref.setdefault(ref, []).append([v0, v1, v2])
                i += 1
            continue

        i += 1

    verts_np = np.array(vertices, dtype=np.float64)
    tri_np = {ref: np.array(tris, dtype=np.int64) for ref, tris in tri_by_ref.items()}
    return verts_np, tri_np


# ---------------------------------------------------------------------------
# Surface classification  (identical to check_surface_verts.py)
# ---------------------------------------------------------------------------

def split_boundary_geometry(
    verts: np.ndarray,
    tris: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    if len(tris) == 0:
        return {}, {}

    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norm  = np.linalg.norm(cross, axis=1)
    nz    = np.zeros(len(tris))
    safe  = norm > 1e-14
    nz[safe] = cross[safe, 2] / norm[safe]

    cent = (v0 + v1 + v2) / 3.0
    cx, cy, cz = cent[:, 0], cent[:, 1], cent[:, 2]
    R = np.sqrt(cx**2 + cy**2)

    all_idx = np.unique(tris.ravel())
    v_all   = verts[all_idx]
    R_all   = np.sqrt(v_all[:, 0]**2 + v_all[:, 1]**2)
    z_all   = v_all[:, 2]

    z_mid = 0.5 * (float(np.min(z_all)) + float(np.max(z_all)))
    r_mid = 0.5 * (float(np.min(R_all)) + float(np.max(R_all)))

    horizontal = np.abs(nz) > 0.7
    top_mask   = horizontal & (cz >= z_mid)
    bot_mask   = horizontal & ~top_mask
    cyl_mask   = ~horizontal

    n_rad = np.zeros(len(tris))
    safe_r = R > 1e-14
    n_rad[safe_r] = (
        cross[safe_r, 0] * cx[safe_r] + cross[safe_r, 1] * cy[safe_r]
    ) / (norm[safe_r] * R[safe_r])

    outer_mask = cyl_mask & ((n_rad >= 0.0) | (R >= r_mid))
    inner_mask = cyl_mask & ~outer_mask

    split_masks = [
        (1001, "inner_wall",  inner_mask),
        (1002, "outer_wall",  outer_mask),
        (1003, "bottom_cap",  bot_mask),
        (1004, "top_cap",     top_mask),
    ]

    tri_by_ref_new: dict[int, np.ndarray] = {}
    labels:         dict[int, str]        = {}
    for ref, label, mask in split_masks:
        idx = np.where(mask)[0]
        if len(idx):
            tri_by_ref_new[ref] = tris[idx]
            labels[ref]         = label

    return tri_by_ref_new, labels


# ---------------------------------------------------------------------------
# Per-mesh deviation computation
# ---------------------------------------------------------------------------

SURFACES = ("inner_wall", "outer_wall")


def mesh_cylindrical_deviations(
    path: Path,
    r_outer: float | None,
    r_inner: float | None,
) -> dict[str, np.ndarray | None]:
    """Return per-vertex |R_xy - R_expected| separately for each cylindrical
    surface.  Keys are 'inner_wall' and 'outer_wall'; value is None when that
    surface was not detected in the mesh."""

    result: dict[str, np.ndarray | None] = {s: None for s in SURFACES}

    verts, tri_by_ref = read_medit(path)
    if not tri_by_ref:
        return result

    all_tris = np.vstack(list(tri_by_ref.values()))
    geo_groups, labels = split_boundary_geometry(verts, all_tris)

    for ref, tris in geo_groups.items():
        label = labels[ref]
        if label not in SURFACES:
            continue

        unique_idx = np.unique(tris.ravel())
        Rxy = np.sqrt(verts[unique_idx, 0]**2 + verts[unique_idx, 1]**2)

        if label == "outer_wall" and r_outer is not None:
            R_exp = r_outer
        elif label == "inner_wall" and r_inner is not None:
            R_exp = r_inner
        else:
            R_exp = float(np.mean(Rxy))

        result[label] = np.abs(Rxy - R_exp)

    return result


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _mesh_sort_key(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 0


def collect_tol_dirs(base: Path) -> list[tuple[float, Path]]:
    """Return list of (tol_value, tol_dir) sorted by tol_value descending."""
    result = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        m = re.fullmatch(r"tol_([0-9.eE+\-]+)", d.name)
        if m:
            result.append((float(m.group(1)), d))
    result.sort(key=lambda x: x[0], reverse=True)
    return result


def collect_meshes(tol_dir: Path) -> list[Path]:
    """Return .mesh files from tol_dir/meshes/, sorted by iteration number."""
    meshes_dir = tol_dir / "meshes"
    if not meshes_dir.is_dir():
        return []
    files = sorted(
        [p for p in meshes_dir.glob("*.mesh")],
        key=_mesh_sort_key,
    )
    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _output_path_for_surface(base_path: Path, surface: str) -> Path:
    """Derive per-surface output path, e.g. out.png → out_outer_wall.png."""
    return base_path.with_name(base_path.stem + f"_{surface}" + base_path.suffix)


def _make_figure(
    surface: str,
    series: dict[float, dict],
    hausd: float,
    out: Path,
) -> None:
    """Draw and save one convergence figure for a single surface."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for tol, data in series.items():
        x     = data["iters"]
        y     = data["mean"]
        y_std = data["std"]

        (line,) = ax.plot(x, y, marker="o", markersize=4, linewidth=1.5,
                          label=f"tol = {tol:g}")
        ax.fill_between(x, y - y_std, y + y_std,
                        alpha=0.15, color=line.get_color())

    ax.axhline(hausd, color="black", linestyle="--", linewidth=1.2,
               label=f"hausd = {hausd:g}")

    title = surface.replace("_", " ").title()
    ax.set_xlabel("Iteration", fontsize=12)
   # ax.set_ylabel("Mean deviation from expected $R$", fontsize=12)
   # ax.set_title(f"{title} — deviation vs. iteration", fontsize=13)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xlim(left=0)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400)
    plt.close(fig)
    print(f"  Saved → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot mean cylindrical-surface deviation vs. iteration for "
                    "each tolerance found in RESULTS_DIR.  Produces one figure "
                    "for the outer wall and one for the inner wall.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("results_dir", metavar="RESULTS_DIR",
                        help="Directory containing tol_*/meshes/ sub-trees.")
    parser.add_argument("output_fig",  metavar="OUTPUT_FIG",
                        help="Base output path; surface name is appended before "
                             "the extension (e.g. out.png → out_outer_wall.png).")
    parser.add_argument("--hausd",   type=float, default=5.0, metavar="VAL",
                        help="Hausdorff tolerance (drawn as a horizontal reference line).")
    parser.add_argument("--r-outer", type=float, default=None, metavar="VAL",
                        help="Expected outer-wall radius. Auto-detected if omitted.")
    parser.add_argument("--r-inner", type=float, default=None, metavar="VAL",
                        help="Expected inner-wall radius. Auto-detected if omitted.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-iteration statistics.")
    args = parser.parse_args()

    base = Path(args.results_dir)
    if not base.is_dir():
        print(f"ERROR: {base} is not a directory.", file=sys.stderr)
        sys.exit(1)

    tol_dirs = collect_tol_dirs(base)
    if not tol_dirs:
        print(f"ERROR: No tol_*/ subdirectories found in {base}.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(tol_dirs)} tolerance(s): {[t for t, _ in tol_dirs]}")

    # ── collect data per surface ─────────────────────────────────────────────
    # series[surface][tol] = {"iters": ndarray, "mean": ndarray, "std": ndarray}
    series: dict[str, dict[float, dict]] = {s: {} for s in SURFACES}

    for tol, tol_dir in tol_dirs:
        meshes = collect_meshes(tol_dir)
        if not meshes:
            print(f"  [tol={tol}] No meshes found in {tol_dir / 'meshes'} — skipped.")
            continue

        # per-surface accumulators
        acc: dict[str, tuple[list, list, list]] = {s: ([], [], []) for s in SURFACES}

        for it, mesh_path in enumerate(meshes):
            devs_by_surface = mesh_cylindrical_deviations(
                mesh_path, args.r_outer, args.r_inner
            )

            for surface in SURFACES:
                devs = devs_by_surface[surface]
                if devs is None or len(devs) == 0:
                    if args.verbose:
                        print(f"  [tol={tol}] iter {it}: "
                              f"{surface} not found — skipped.")
                    continue

                mean_dev = float(np.mean(devs))
                std_dev  = float(np.std(devs))
                iters_l, means_l, stds_l = acc[surface]
                iters_l.append(it)
                means_l.append(mean_dev)
                stds_l.append(std_dev)

                if args.verbose:
                    print(f"  [tol={tol}] iter {it:3d}  {surface}  "
                          f"n={len(devs):6d}  "
                          f"mean={mean_dev:.4e}  std={std_dev:.4e}  "
                          f"max={float(np.max(devs)):.4e}  ({mesh_path.name})")

        for surface in SURFACES:
            iters_l, means_l, stds_l = acc[surface]
            if iters_l:
                series[surface][tol] = {
                    "iters": np.array(iters_l),
                    "mean":  np.array(means_l),
                    "std":   np.array(stds_l),
                }

    # ── produce one figure per surface ───────────────────────────────────────
    out_base = Path(args.output_fig)
    any_saved = False

    for surface in SURFACES:
        if not series[surface]:
            print(f"\n[{surface}] No data — figure skipped.")
            continue
        out = _output_path_for_surface(out_base, surface)
        print(f"\n[{surface}]")
        _make_figure(surface, series[surface], args.hausd, out)
        any_saved = True

    if not any_saved:
        print("ERROR: No figures produced — no cylindrical surface data found.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
