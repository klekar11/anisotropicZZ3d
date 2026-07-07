"""relabel_tok_mesh.py
======================
Read a tokamak Medit .mesh file where ALL boundary triangles carry ref=0,
classify them geometrically into inner wall / outer wall / bottom cap / top cap,
assign distinct non-zero reference tags, and write a corrected .mesh file that
MMG3D can use for per-patch Hausdorff / hmin / hmax control.

Output reference convention (matches the vertex refs already in the file):
    REF_INNER  = 2   inner cylindrical wall
    REF_OUTER  = 3   outer cylindrical wall
    REF_BOTTOM = 4   bottom horizontal cap  (min z)
    REF_TOP    = 5   top    horizontal cap  (max z)

Optionally also appends a RequiredTriangles section to freeze the boundary
patches completely (equivalent to -nosurf but patch-selective).

Usage
-----
    python relabel_tok_mesh.py input.mesh output.mesh [OPTIONS]

    --freeze-boundary     Append RequiredTriangles for ALL boundary triangles.
    --freeze-inner        Append RequiredTriangles for the inner wall only.
    --freeze-outer        Append RequiredTriangles for the outer wall only.
    --nz-threshold FLOAT  |n_z| threshold to classify horizontal triangles.
                          Default: 0.7
    --verbose             Print classification statistics.

After running this script you can use a .mmg3d parameter file to set
per-patch Hausdorff distances, e.g.:

    # mycase.mmg3d
    LST
    parameters
    4
    2 Triangles  5.0  2.0  50.0   # inner wall:  hausd=5   hmin=2   hmax=50
    3 Triangles  5.0  2.0  50.0   # outer wall:  hausd=5   hmin=2   hmax=50
    4 Triangles  1e9  2.0 200.0   # bottom cap:  hausd=huge (don't touch shape)
    5 Triangles  1e9  2.0 200.0   # top cap:     hausd=huge
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Output reference tags
# ---------------------------------------------------------------------------
REF_INNER  = 2
REF_OUTER  = 3
REF_BOTTOM = 4
REF_TOP    = 5

LABEL_MAP = {
    REF_INNER:  "inner_wall",
    REF_OUTER:  "outer_wall",
    REF_BOTTOM: "bottom_cap",
    REF_TOP:    "top_cap",
}


# ---------------------------------------------------------------------------
# Medit parser  (reads ALL sections verbatim for faithful round-trip)
# ---------------------------------------------------------------------------

class MeditMesh:
    """Container for a parsed Medit .mesh file."""

    def __init__(self) -> None:
        self.version: int = 2
        self.dim: int = 3
        # Each entry: list of raw token strings per line (excluding keyword)
        self.vertices:     list[list[str]] = []
        self.triangles:    list[list[str]] = []  # v0 v1 v2 ref  (1-based, str)
        self.tetrahedra:   list[list[str]] = []
        self.edges:        list[list[str]] = []
        self.extra_sections: list[tuple[str, list[str]]] = []  # (keyword, raw_lines)

    @property
    def n_verts(self) -> int:
        return len(self.vertices)

    def vertex_coords(self) -> np.ndarray:
        """Return (N,3) float array of vertex coordinates."""
        return np.array([[float(v[0]), float(v[1]), float(v[2])]
                         for v in self.vertices], dtype=np.float64)


def parse_medit(path: Path) -> MeditMesh:
    mesh = MeditMesh()

    with open(path, "r") as f:
        lines = [l.rstrip() for l in f]

    i = 0
    n = len(lines)

    known_sections = {
        "Vertices", "Triangles", "Tetrahedra", "Edges",
        "RequiredVertices", "RequiredTriangles", "RequiredEdges",
        "RequiredTetrahedra", "Corners", "Ridges", "End",
    }

    def next_nonempty() -> str | None:
        nonlocal i
        while i < n:
            s = lines[i].strip()
            if s and not s.startswith("#"):
                return s
            i += 1
        return None

    while i < n:
        tok = next_nonempty()
        if tok is None:
            break

        # ---- Header ----
        if tok.startswith("MeshVersionFormatted"):
            parts = tok.split()
            if len(parts) > 1:
                mesh.version = int(parts[1])
            i += 1
            continue

        if tok.startswith("Dimension"):
            parts = tok.split()
            if len(parts) > 1:
                mesh.dim = int(parts[1])
                i += 1
            else:
                i += 1
                t2 = next_nonempty()
                if t2:
                    mesh.dim = int(t2)
                i += 1
            continue

        if tok == "End":
            i += 1
            break

        # ---- Vertices ----
        if tok == "Vertices":
            i += 1
            count_str = next_nonempty()
            i += 1
            count = int(count_str)
            for _ in range(count):
                s = next_nonempty()
                i += 1
                mesh.vertices.append(s.split())
            continue

        # ---- Triangles ----
        if tok == "Triangles":
            i += 1
            count_str = next_nonempty()
            i += 1
            count = int(count_str)
            for _ in range(count):
                s = next_nonempty()
                i += 1
                mesh.triangles.append(s.split())
            continue

        # ---- Tetrahedra ----
        if tok == "Tetrahedra":
            i += 1
            count_str = next_nonempty()
            i += 1
            count = int(count_str)
            for _ in range(count):
                s = next_nonempty()
                i += 1
                mesh.tetrahedra.append(s.split())
            continue

        # ---- Edges ----
        if tok == "Edges":
            i += 1
            count_str = next_nonempty()
            i += 1
            count = int(count_str)
            for _ in range(count):
                s = next_nonempty()
                i += 1
                mesh.edges.append(s.split())
            continue

        # ---- Required* and other known sections: read count + indices ----
        if tok in known_sections:
            i += 1
            count_str = next_nonempty()
            i += 1
            count = int(count_str) if count_str else 0
            raw: list[str] = []
            for _ in range(count):
                s = next_nonempty()
                i += 1
                raw.append(s)
            mesh.extra_sections.append((tok, raw))
            continue

        # Unknown token — skip
        i += 1

    return mesh


# ---------------------------------------------------------------------------
# Geometric classification of boundary triangles
# ---------------------------------------------------------------------------

def classify_triangles(
    verts: np.ndarray,
    triangles: list[list[str]],
    nz_threshold: float = 0.7,
) -> np.ndarray:
    """Return an int array of length len(triangles) with new ref tags.

    Classification uses the face normal and centroid radius, exactly as in
    check_surface_verts.py's split_boundary_geometry.
    """
    n = len(triangles)
    new_refs = np.zeros(n, dtype=np.int32)

    # Build triangle arrays (0-based indices)
    tri_idx = np.array([[int(t[0]) - 1, int(t[1]) - 1, int(t[2]) - 1]
                        for t in triangles], dtype=np.int64)

    v0 = verts[tri_idx[:, 0]]
    v1 = verts[tri_idx[:, 1]]
    v2 = verts[tri_idx[:, 2]]

    cross   = np.cross(v1 - v0, v2 - v0)          # (n, 3)
    norm_c  = np.linalg.norm(cross, axis=1)        # (n,)
    nz      = np.zeros(n, dtype=np.float64)
    safe    = norm_c > 1e-14
    nz[safe] = cross[safe, 2] / norm_c[safe]

    cent = (v0 + v1 + v2) / 3.0
    cx, cy, cz = cent[:, 0], cent[:, 1], cent[:, 2]
    R = np.sqrt(cx**2 + cy**2)

    # Global z and R midpoints to separate top/bottom and inner/outer
    all_idx = np.unique(tri_idx.ravel())
    v_all   = verts[all_idx]
    R_all   = np.sqrt(v_all[:, 0]**2 + v_all[:, 1]**2)
    z_all   = v_all[:, 2]
    z_mid   = 0.5 * (float(np.min(z_all)) + float(np.max(z_all)))
    r_mid   = 0.5 * (float(np.min(R_all)) + float(np.max(R_all)))

    horizontal = np.abs(nz) > nz_threshold
    top_mask   = horizontal & (cz >= z_mid)
    bot_mask   = horizontal & ~top_mask
    cyl_mask   = ~horizontal

    # Radial component of normal to separate inner/outer
    n_rad = np.zeros(n, dtype=np.float64)
    safe_r = R > 1e-14
    n_rad[safe_r] = (
        cross[safe_r, 0] * cx[safe_r] + cross[safe_r, 1] * cy[safe_r]
    ) / (norm_c[safe_r] * R[safe_r])

    outer_mask = cyl_mask & ((n_rad >= 0.0) | (R >= r_mid))
    inner_mask = cyl_mask & ~outer_mask

    new_refs[inner_mask] = REF_INNER
    new_refs[outer_mask] = REF_OUTER
    new_refs[bot_mask]   = REF_BOTTOM
    new_refs[top_mask]   = REF_TOP

    # Any unclassified triangle (degenerate?) gets ref=1 as a fallback
    unclassified = new_refs == 0
    new_refs[unclassified] = 1

    return new_refs


# ---------------------------------------------------------------------------
# Medit writer
# ---------------------------------------------------------------------------

def write_medit(
    mesh: MeditMesh,
    new_refs: np.ndarray,
    path: Path,
    required_mask: np.ndarray | None = None,
) -> None:
    """Write the corrected .mesh file.

    Parameters
    ----------
    mesh :
        Parsed mesh object.
    new_refs :
        Array of length len(mesh.triangles) with the new reference tags.
    path :
        Output file path.
    required_mask :
        Boolean array of length len(mesh.triangles). Triangles where this is
        True will be listed in a RequiredTriangles section.
    """
    with open(path, "w") as f:
        f.write(f"MeshVersionFormatted {mesh.version}\n")
        f.write(f" Dimension\n {mesh.dim}\n")

        # Vertices (unchanged)
        f.write(f" Vertices\n {mesh.n_verts}\n")
        for v in mesh.vertices:
            f.write(" " + "  ".join(v) + "\n")

        # Triangles with updated refs
        f.write(f" Triangles\n {len(mesh.triangles)}\n")
        for k, t in enumerate(mesh.triangles):
            # t = [v0, v1, v2, old_ref]  — replace last token
            f.write(f" {t[0]} {t[1]} {t[2]} {new_refs[k]}\n")

        # RequiredTriangles (optional)
        if required_mask is not None:
            req_indices = np.where(required_mask)[0] + 1  # 1-based
            f.write(f" RequiredTriangles\n {len(req_indices)}\n")
            for idx in req_indices:
                f.write(f" {idx}\n")

        # Tetrahedra (unchanged)
        if mesh.tetrahedra:
            f.write(f" Tetrahedra\n {len(mesh.tetrahedra)}\n")
            for t in mesh.tetrahedra:
                f.write(" " + "  ".join(t) + "\n")

        # Edges (unchanged)
        if mesh.edges:
            f.write(f" Edges\n {len(mesh.edges)}\n")
            for e in mesh.edges:
                f.write(" " + "  ".join(e) + "\n")

        # Pass through any other sections (Corners, Ridges, etc.)
        # — but drop any pre-existing RequiredTriangles since we rewrite it
        for kw, raw in mesh.extra_sections:
            if kw == "RequiredTriangles":
                continue   # we wrote our own above
            f.write(f" {kw}\n {len(raw)}\n")
            for line in raw:
                f.write(f" {line}\n")

        f.write(" End\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-label tokamak Medit boundary triangles with distinct "
            "non-zero refs so MMG3D can apply per-patch parameters."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input",  help="Input  .mesh file (all-zero triangle refs).")
    parser.add_argument("output", help="Output .mesh file with corrected refs.")
    parser.add_argument(
        "--freeze-boundary",
        action="store_true",
        help="Append RequiredTriangles for ALL boundary triangles.",
    )
    parser.add_argument(
        "--freeze-inner",
        action="store_true",
        help="Append RequiredTriangles for the inner wall only.",
    )
    parser.add_argument(
        "--freeze-outer",
        action="store_true",
        help="Append RequiredTriangles for the outer wall only.",
    )
    parser.add_argument(
        "--nz-threshold",
        type=float,
        default=0.7,
        metavar="FLOAT",
        help="|n_z| threshold above which a triangle is considered horizontal.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print classification statistics.",
    )

    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # ---- Parse ----
    print(f"Reading  {in_path} ...", end=" ", flush=True)
    mesh = parse_medit(in_path)
    print(f"done  ({mesh.n_verts} verts, "
          f"{len(mesh.triangles)} tris, "
          f"{len(mesh.tetrahedra)} tets)")

    if not mesh.triangles:
        print("ERROR: no triangles found in the mesh.", file=sys.stderr)
        sys.exit(1)

    # ---- Sanity check on input refs ----
    old_refs = np.array([int(t[3]) for t in mesh.triangles], dtype=np.int32)
    unique_old = np.unique(old_refs)
    if args.verbose or not np.all(old_refs == 0):
        print(f"  Input triangle refs: {unique_old.tolist()}")
        if not np.all(old_refs == 0):
            print("  WARNING: not all refs are 0 — relabelling will overwrite them.")

    # ---- Classify ----
    verts   = mesh.vertex_coords()
    new_refs = classify_triangles(verts, mesh.triangles, args.nz_threshold)

    if args.verbose:
        for ref, label in LABEL_MAP.items():
            count = int(np.sum(new_refs == ref))
            v_idx = np.unique(np.array([
                [int(t[0]) - 1, int(t[1]) - 1, int(t[2]) - 1]
                for t, r in zip(mesh.triangles, new_refs) if r == ref
            ]).ravel()) if count else np.array([])
            if count:
                v = verts[v_idx]
                Rxy = np.sqrt(v[:, 0]**2 + v[:, 1]**2)
                print(f"  [{label:12s}] ref={ref}  "
                      f"n_tris={count:5d}  "
                      f"mean_Rxy={np.mean(Rxy):8.2f}  "
                      f"std_Rxy={np.std(Rxy):7.3f}  "
                      f"z=[{np.min(v[:,2]):.1f}, {np.max(v[:,2]):.1f}]")

        unclassified = int(np.sum(new_refs == 1))
        if unclassified:
            print(f"  [unclassified   ] ref=1   n_tris={unclassified}")

    # ---- Build RequiredTriangles mask ----
    required_mask = None
    if args.freeze_boundary:
        required_mask = np.ones(len(mesh.triangles), dtype=bool)
    elif args.freeze_inner or args.freeze_outer:
        required_mask = np.zeros(len(mesh.triangles), dtype=bool)
        if args.freeze_inner:
            required_mask |= (new_refs == REF_INNER)
        if args.freeze_outer:
            required_mask |= (new_refs == REF_OUTER)

    n_req = int(np.sum(required_mask)) if required_mask is not None else 0

    # ---- Write ----
    print(f"Writing  {out_path} ...", end=" ", flush=True)
    write_medit(mesh, new_refs, out_path, required_mask)
    print("done")

    print(f"\nSummary:")
    print(f"  Inner wall  (ref={REF_INNER}): {int(np.sum(new_refs == REF_INNER)):5d} triangles")
    print(f"  Outer wall  (ref={REF_OUTER}): {int(np.sum(new_refs == REF_OUTER)):5d} triangles")
    print(f"  Bottom cap  (ref={REF_BOTTOM}): {int(np.sum(new_refs == REF_BOTTOM)):5d} triangles")
    print(f"  Top cap     (ref={REF_TOP}): {int(np.sum(new_refs == REF_TOP)):5d} triangles")
    if n_req:
        print(f"  RequiredTriangles written: {n_req}")

    # ---- Print per-patch .mmg3d parameter file snippet ----
    print(f"""
To use per-patch Hausdorff in MMG, create a file named
  {out_path.stem}.mmg3d
with content:

  LST
  parameters
  4
  {REF_INNER} Triangles  <hausd>  <hmin>  <hmax>   # inner wall
  {REF_OUTER} Triangles  <hausd>  <hmin>  <hmax>   # outer wall
  {REF_BOTTOM} Triangles  1e9     <hmin>  <hmax>   # bottom cap (frozen shape)
  {REF_TOP} Triangles  1e9        <hmin>  <hmax>   # top cap   (frozen shape)

Replace <hausd>, <hmin>, <hmax> with your values (in mm).
Then run MMG without -nosurf and without -hausd:
  mmg3d -in {out_path} -sol metric.sol -out adapted.mesh
""")


if __name__ == "__main__":
    main()
