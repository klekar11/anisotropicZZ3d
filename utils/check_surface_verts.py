"""check_tok_boundary.py
=======================
Verify that boundary vertices of a tokamak Medit mesh respect the prescribed
Hausdorff tolerance against the ideal cylindrical surfaces (inner and outer
walls).  Top and bottom caps are detected and reported separately but are not
checked against a circle.

Usage
-----
    python check_tok_boundary.py mesh1.mesh [mesh2.mesh ...] [OPTIONS]

Minimal example (auto-detect all radii from the mesh):
    python check_tok_boundary.py mesh_5.mesh --hausd 5.0

With explicit expected radii:
    python check_tok_boundary.py mesh_*.mesh \\
        --hausd 5.0          \\
        --r-outer 875.0      \\
        --r-inner 405.0      \\
        --verbose

Options
-------
--hausd VAL      Hausdorff tolerance used in the MMG run (mm).  Default 5.0.
--r-outer VAL    Expected outer-wall cylindrical radius (mm).  If omitted the
                 mean detected radius of the outer group is used as reference.
--r-inner VAL    Expected inner-wall cylindrical radius (mm).  Same fallback.
--verbose        Print per-vertex details for vertices outside the tolerance.
--csv PATH       Write a CSV summary (one row per mesh × surface group).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Medit parser
# ---------------------------------------------------------------------------

def read_medit(path: Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Parse a Medit .mesh file.

    Returns
    -------
    vertices : ndarray, shape (N, 3)
        Vertex coordinates.
    tri_by_ref : dict  {ref_tag -> ndarray of shape (M, 3), vertex indices 0-based}
        Boundary triangles grouped by their reference tag.
    """
    vertices: list[list[float]] = []
    tri_by_ref: dict[int, list[list[int]]] = {}

    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    n = len(lines)

    def skip_blank_and_comments() -> None:
        nonlocal i
        while i < n and (lines[i].strip() == "" or lines[i].strip().startswith("#")):
            i += 1

    while i < n:
        token = lines[i].strip().split()
        if not token:
            i += 1
            continue

        keyword = token[0]

        # ---- Vertices ----
        if keyword == "Vertices":
            i += 1
            skip_blank_and_comments()
            count = int(lines[i].strip())
            i += 1
            for _ in range(count):
                skip_blank_and_comments()
                parts = lines[i].strip().split()
                # x y z ref  (ref is ignored for vertices)
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                i += 1
            continue

        # ---- Triangles (boundary faces) ----
        if keyword == "Triangles":
            i += 1
            skip_blank_and_comments()
            count = int(lines[i].strip())
            i += 1
            for _ in range(count):
                skip_blank_and_comments()
                parts = lines[i].strip().split()
                # v0 v1 v2 ref   (1-based vertex indices)
                v0, v1, v2 = int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1
                ref = int(parts[3])
                tri_by_ref.setdefault(ref, []).append([v0, v1, v2])
                i += 1
            continue

        i += 1

    verts_np = np.array(vertices, dtype=np.float64)
    tri_np = {ref: np.array(tris, dtype=np.int64)
              for ref, tris in tri_by_ref.items()}
    return verts_np, tri_np


# ---------------------------------------------------------------------------
# Surface classification
# ---------------------------------------------------------------------------

class SurfaceKind:
    OUTER_WALL   = "outer_wall"
    INNER_WALL   = "inner_wall"
    TOP_CAP      = "top_cap"
    BOTTOM_CAP   = "bottom_cap"
    UNKNOWN      = "unknown"


def split_boundary_geometry(
    verts: np.ndarray,
    tris: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    """Split boundary triangles into geometric sub-groups.

    This is independent of original Medit/MMG reference tags and is robust when
    all boundary faces are emitted under a single reference tag.
    """
    if len(tris) == 0:
        return {}, {}

    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(cross, axis=1)
    nz = np.zeros(len(tris), dtype=np.float64)
    safe = norm > 1e-14
    nz[safe] = cross[safe, 2] / norm[safe]

    cent = (v0 + v1 + v2) / 3.0
    cx = cent[:, 0]
    cy = cent[:, 1]
    cz = cent[:, 2]
    R = np.sqrt(cx**2 + cy**2)

    all_idx = np.unique(tris.ravel())
    v_all = verts[all_idx]
    R_all = np.sqrt(v_all[:, 0]**2 + v_all[:, 1]**2)
    z_all = v_all[:, 2]

    z_mid = 0.5 * (float(np.min(z_all)) + float(np.max(z_all)))
    r_mid = 0.5 * (float(np.min(R_all)) + float(np.max(R_all)))

    # Strong |n_z| means near-horizontal triangle.
    horizontal = np.abs(nz) > 0.7
    top_mask = horizontal & (cz >= z_mid)
    bot_mask = horizontal & ~top_mask

    cyl_mask = ~horizontal
    n_rad = np.zeros(len(tris), dtype=np.float64)
    safe_r = R > 1e-14
    n_rad[safe_r] = (cross[safe_r, 0] * cx[safe_r] + cross[safe_r, 1] * cy[safe_r]) / (
        norm[safe_r] * R[safe_r]
    )

    # Prefer normal sign for inner/outer walls; fall back to centroid radius.
    outer_mask = cyl_mask & ((n_rad >= 0.0) | (R >= r_mid))
    inner_mask = cyl_mask & ~outer_mask

    split_masks: list[tuple[int, str, np.ndarray]] = [
        (1001, SurfaceKind.INNER_WALL, inner_mask),
        (1002, SurfaceKind.OUTER_WALL, outer_mask),
        (1003, SurfaceKind.BOTTOM_CAP, bot_mask),
        (1004, SurfaceKind.TOP_CAP, top_mask),
    ]

    tri_by_ref_new: dict[int, np.ndarray] = {}
    forced_labels: dict[int, str] = {}
    for ref, label, mask in split_masks:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        tri_by_ref_new[ref] = tris[idx]
        forced_labels[ref] = label

    return tri_by_ref_new, forced_labels


# ---------------------------------------------------------------------------
# Per-group geometry statistics
# ---------------------------------------------------------------------------

def group_stats(
    verts: np.ndarray,
    tris: np.ndarray,
) -> dict:
    """Return geometric statistics for a boundary group.

    Uses triangle-area weighting so that the statistics represent the
    surface rather than the vertex sampling density.
    """
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]

    # Triangle areas (unnormalised)
    cross  = np.cross(v1 - v0, v2 - v0)
    areas  = 0.5 * np.linalg.norm(cross, axis=1)   # (n_tri,)
    total_area = float(np.sum(areas))

    # Unique vertex indices for this group
    unique_idx = np.unique(tris.ravel())
    v_group    = verts[unique_idx]

    Rxy = np.sqrt(v_group[:, 0]**2 + v_group[:, 1]**2)
    z   = v_group[:, 2]

    return {
        "n_vertices": len(unique_idx),
        "n_triangles": len(tris),
        "total_area": total_area,
        "mean_Rxy": float(np.mean(Rxy)),
        "std_Rxy": float(np.std(Rxy)),
        "min_Rxy": float(np.min(Rxy)),
        "max_Rxy": float(np.max(Rxy)),
        "mean_z": float(np.mean(z)),
        "std_z": float(np.std(z)),
        "min_z": float(np.min(z)),
        "max_z": float(np.max(z)),
        "unique_idx": unique_idx,
    }


# ---------------------------------------------------------------------------
# Hausdorff check for cylindrical surfaces
# ---------------------------------------------------------------------------

def check_cylindrical(
    verts: np.ndarray,
    unique_idx: np.ndarray,
    R_expected: float,
    hausd: float,
    label: str,
    verbose: bool = False,
) -> dict:
    """Check how close boundary vertices are to the cylinder R_xy = R_expected.

    The Hausdorff distance from a point (x, y, z) to the ideal infinite
    cylinder of radius R_expected centred on the z-axis is simply:
        dist = |R_xy - R_expected|

    Returns
    -------
    result dict with keys: ok, max_dev, mean_dev, n_outside, fraction_outside
    """
    v = verts[unique_idx]
    Rxy   = np.sqrt(v[:, 0]**2 + v[:, 1]**2)
    devs  = np.abs(Rxy - R_expected)

    max_dev   = float(np.max(devs))
    mean_dev  = float(np.mean(devs))
    n_outside = int(np.sum(devs > hausd))
    frac_out  = n_outside / len(devs)
    ok        = n_outside == 0

    if verbose and n_outside > 0:
        print(f"\n  [{label}] Vertices outside hausd={hausd:.4g} mm "
              f"(R_expected={R_expected:.4g} mm):")
        bad = np.where(devs > hausd)[0]
        for bi in bad[:20]:   # cap at 20 lines
            gi = unique_idx[bi]
            x, y, z = v[bi]
            print(f"    vertex {gi:6d}  "
                  f"x={x:10.4f}  y={y:10.4f}  z={z:10.4f}  "
                  f"Rxy={Rxy[bi]:10.4f}  dev={devs[bi]:.4e}")
        if len(bad) > 20:
            print(f"    ... and {len(bad) - 20} more")

    return {
        "ok": ok,
        "max_dev": max_dev,
        "mean_dev": mean_dev,
        "n_vertices": len(unique_idx),
        "n_outside": n_outside,
        "fraction_outside": frac_out,
    }


# ---------------------------------------------------------------------------
# Main check for a single mesh file
# ---------------------------------------------------------------------------

def check_mesh(
    path: Path,
    hausd: float,
    r_outer_expected: float | None,
    r_inner_expected: float | None,
    verbose: bool,
) -> list[dict]:
    """Run all checks on one mesh file.  Returns a list of result dicts."""

    print(f"\n{'='*70}")
    print(f"  Mesh: {path.name}")
    print(f"{'='*70}")

    verts, tri_by_ref = read_medit(path)
    n_verts_total = len(verts)
    n_tris_total  = sum(len(t) for t in tri_by_ref.values())

    print(f"  Total vertices : {n_verts_total}")
    print(f"  Boundary tags  : {sorted(tri_by_ref.keys())}")
    print(f"  Boundary tris  : {n_tris_total}")

    if not tri_by_ref:
        print("  WARNING: No boundary triangles found.")
        return []

    all_boundary_tris = np.vstack(list(tri_by_ref.values()))
    tri_by_ref_geom, labels = split_boundary_geometry(verts, all_boundary_tris)

    if len(tri_by_ref_geom) < 2:
        print("  WARNING: Could not split boundary into geometric surfaces.")
        return []

    print("  NOTE: Geometry-based split produced pseudo-groups:")
    print(f"        {sorted((ref, labels[ref]) for ref in tri_by_ref_geom.keys())}")

    # ---- Group statistics by geometric pseudo-reference ----
    cylindrical_groups: list[tuple[int, dict]] = []  # (ref, stats)
    horizontal_groups:  list[tuple[int, dict]] = []

    for ref, tris in tri_by_ref_geom.items():
        stats   = group_stats(verts, tris)
        label = labels.get(ref, SurfaceKind.UNKNOWN)
        stats["label"] = label
        stats["ref"]      = ref

        if label in (SurfaceKind.INNER_WALL, SurfaceKind.OUTER_WALL):
            cylindrical_groups.append((ref, stats))
        else:
            horizontal_groups.append((ref, stats))

    if len(cylindrical_groups) == 0:
        print("  WARNING: No cylindrical surface groups detected after geometry split.")
    elif len(cylindrical_groups) == 1:
        print("  WARNING: Only one cylindrical group detected "
              "(expected at least 2: inner + outer).")

    # ---- Run checks ----
    results: list[dict] = []

    # Cylindrical groups
    for ref, stats in cylindrical_groups:
        label = stats["label"]

        # Decide expected radius
        if label == SurfaceKind.OUTER_WALL and r_outer_expected is not None:
            R_exp = r_outer_expected
        elif label == SurfaceKind.INNER_WALL and r_inner_expected is not None:
            R_exp = r_inner_expected
        else:
            R_exp = stats["mean_Rxy"]   # fall back to detected mean

        check = check_cylindrical(
            verts, stats["unique_idx"], R_exp, hausd, label, verbose
        )

        status = "PASS" if check["ok"] else "FAIL"
        print(f"\n  [{label}]  ref={ref}  "
              f"n_verts={stats['n_vertices']}  "
              f"R_exp={R_exp:.4g} mm")
        print(f"    mean(Rxy)={stats['mean_Rxy']:.4g}  "
              f"std(Rxy)={stats['std_Rxy']:.4g}  "
              f"std(z)={stats['std_z']:.4g}")
        print(f"    max_dev={check['max_dev']:.4e}  "
              f"mean_dev={check['mean_dev']:.4e}  "
              f"hausd={hausd:.4g}  "
              f"n_outside={check['n_outside']}  "
              f"→ {status}")

        results.append({
            "mesh":              path.name,
            "ref":               ref,
            "label":             label,
            "n_vertices":        check["n_vertices"],
            "R_expected":        R_exp,
            "mean_Rxy":          stats["mean_Rxy"],
            "std_Rxy":           stats["std_Rxy"],
            "max_dev":           check["max_dev"],
            "mean_dev":          check["mean_dev"],
            "hausd":             hausd,
            "n_outside":         check["n_outside"],
            "fraction_outside":  check["fraction_outside"],
            "status":            status,
        })

    # Horizontal groups (informational only)
    for ref, stats in horizontal_groups:
        label = stats["label"]
        print(f"\n  [{label}]  ref={ref}  "
              f"n_verts={stats['n_vertices']}  "
              f"(not cylindrical — no Hausdorff check)")
        print(f"    z_range=[{stats['min_z']:.4g}, {stats['max_z']:.4g}]  "
              f"mean_z={stats['mean_z']:.4g}  "
              f"Rxy_range=[{stats['min_Rxy']:.4g}, {stats['max_Rxy']:.4g}]")

        results.append({
            "mesh":              path.name,
            "ref":               ref,
            "label":             label,
            "n_vertices":        stats["n_vertices"],
            "R_expected":        float("nan"),
            "mean_Rxy":          stats["mean_Rxy"],
            "std_Rxy":           stats["std_Rxy"],
            "max_dev":           float("nan"),
            "mean_dev":          float("nan"),
            "hausd":             hausd,
            "n_outside":         0,
            "fraction_outside":  0.0,
            "status":            "N/A",
        })

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check tokamak mesh boundary vertices against ideal cylindrical walls.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "meshes",
        nargs="+",
        metavar="MESH",
        help="One or more Medit .mesh files to check.",
    )
    parser.add_argument(
        "--hausd",
        type=float,
        default=5.0,
        metavar="VAL",
        help="Hausdorff tolerance used in the MMG run (mm).",
    )
    parser.add_argument(
        "--r-outer",
        type=float,
        default=None,
        metavar="VAL",
        help="Expected outer-wall cylindrical radius (mm). "
             "If omitted, the mean detected R_xy of the outer group is used.",
    )
    parser.add_argument(
        "--r-inner",
        type=float,
        default=None,
        metavar="VAL",
        help="Expected inner-wall cylindrical radius (mm). "
             "If omitted, the mean detected R_xy of the inner group is used.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print coordinates of every vertex outside the tolerance.",
    )
    parser.add_argument(
        "--csv",
        metavar="PATH",
        default=None,
        help="Write a CSV summary to this path.",
    )

    args = parser.parse_args()

    all_results: list[dict] = []

    for mesh_str in args.meshes:
        path = Path(mesh_str)
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            continue
        results = check_mesh(
            path,
            hausd=args.hausd,
            r_outer_expected=args.r_outer,
            r_inner_expected=args.r_inner,
            verbose=args.verbose,
        )
        all_results.extend(results)

    # ---- Global summary ----
    print(f"\n{'='*70}")
    print("  GLOBAL SUMMARY")
    print(f"{'='*70}")
    cyl_results = [r for r in all_results if r["status"] in ("PASS", "FAIL")]
    if cyl_results:
        n_pass = sum(1 for r in cyl_results if r["status"] == "PASS")
        n_fail = sum(1 for r in cyl_results if r["status"] == "FAIL")
        print(f"  Cylindrical surface checks:  {n_pass} PASS  /  {n_fail} FAIL")
        print(f"\n  {'Mesh':<30} {'Surface':<20} {'R_exp':>8} "
              f"{'max_dev':>10} {'n_out':>6} {'Status':>6}")
        print("  " + "-" * 86)
        for r in cyl_results:
            print(f"  {r['mesh']:<30} {r['label']:<20} {r['R_expected']:>8.4g} "
                  f"{r['max_dev']:>10.4e} {r['n_outside']:>6} {r['status']:>6}")
    else:
        print("  No cylindrical surfaces were checked.")

    # ---- Optional CSV output ----
    if args.csv:
        csv_path = Path(args.csv)
        fieldnames = [
            "mesh", "ref", "label", "n_vertices",
            "R_expected", "mean_Rxy", "std_Rxy",
            "max_dev", "mean_dev", "hausd",
            "n_outside", "fraction_outside", "status",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n  CSV written → {csv_path}")


if __name__ == "__main__":
    main()
