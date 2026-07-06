import argparse
import re
from pathlib import Path


def tol_key(path: Path) -> float:
    m = re.search(r'(\d+\.?\d*(?:e[+-]?\d+)?)', path.name, re.IGNORECASE)
    return float(m.group()) if m else 0.0


def numeric_key(path: Path) -> int:
    m = re.search(r'\d+', path.stem)
    return int(m.group()) if m else 0


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-tolerance VTU sequences into one PVD animation. "
                    "Expects subdirectories named tol_X/, each containing a vtk/ folder with .vtu files."
    )
    parser.add_argument("parent_dir", help="Directory containing tol_X/ subdirectories")
    parser.add_argument("--output", default="combined.pvd", help="Output filename (default: combined.pvd)")
    parser.add_argument("--vtk-subdir", default="vtk",
                        help="Name of the subfolder inside each tol_X/ that holds .vtu files (default: vtk)")
    parser.add_argument("--fine-first", action="store_true",
                        help="Order from finest to coarsest mesh (smallest tol first). "
                             "Default is coarse-to-fine (largest tol first).")
    args = parser.parse_args()

    parent = Path(args.parent_dir).resolve()
    if not parent.is_dir():
        raise NotADirectoryError(f"Not a directory: {parent}")

    # Find tol_X subdirs that contain the vtk subfolder with .vtu files
    tol_dirs = []
    for d in parent.iterdir():
        if not d.is_dir():
            continue
        vtk_dir = d / args.vtk_subdir
        if vtk_dir.is_dir() and list(vtk_dir.glob("*.vtu")):
            tol_dirs.append(d)

    if not tol_dirs:
        print(f"No tol_X/{args.vtk_subdir}/*.vtu structure found under {parent}")
        return

    # Sort by numeric tolerance value: default coarse-to-fine = largest tol first
    tol_dirs.sort(key=tol_key, reverse=not args.fine_first)

    pvd_path = parent / args.output
    timestep = 0
    total_frames = 0

    with open(pvd_path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1">\n')
        f.write('  <Collection>\n')
        for tol_dir in tol_dirs:
            vtk_dir = tol_dir / args.vtk_subdir
            files = sorted(vtk_dir.glob("*.vtu"), key=numeric_key)
            for vtu in files:
                rel = vtu.relative_to(parent)
                f.write(f'    <DataSet timestep="{timestep}" file="{rel}"/>\n')
                timestep += 1
                total_frames += 1

        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')

    print(f"Wrote {pvd_path}")
    print(f"  {len(tol_dirs)} tolerances, {total_frames} total frames")
    print(f"  Order ({'fine→coarse' if args.fine_first else 'coarse→fine'}):")
    for d in tol_dirs:
        n = len(list((d / args.vtk_subdir).glob("*.vtu")))
        print(f"    {d.name}/  ({n} frames)")


if __name__ == "__main__":
    main()
