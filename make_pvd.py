import argparse
import re
from pathlib import Path


def numeric_key(path: Path) -> int:
    m = re.search(r'\d+', path.stem)
    return int(m.group()) if m else 0


def main():
    parser = argparse.ArgumentParser(
        description="Create a ParaView PVD collection file from all .vtu files in a directory."
    )
    parser.add_argument("directory", help="Path to the directory containing .vtu files")
    args = parser.parse_args()

    vtk_dir = Path(args.directory).resolve()
    if not vtk_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {vtk_dir}")

    files = sorted(vtk_dir.glob("*.vtu"), key=numeric_key)

    if not files:
        print(f"No .vtu files found in {vtk_dir}")
        return

    pvd_path = vtk_dir / "animation.pvd"
    with open(pvd_path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1">\n')
        f.write('  <Collection>\n')
        for i, file in enumerate(files):
            f.write(f'    <DataSet timestep="{i}" file="{file.name}"/>\n')
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')

    print(f"Wrote {len(files)} entries → {pvd_path}")


if __name__ == "__main__":
    main()
