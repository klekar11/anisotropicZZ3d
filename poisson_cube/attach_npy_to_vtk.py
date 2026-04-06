#!/usr/bin/env python3
"""Attach nodal solutions stored in .npy files to VTK/VTU meshes.

Examples
--------
Single pair:
  python attach_npy_to_vtk.py --npy path/to/solution.npy --vtk path/to/mesh.vtk

Directory mode (pairs by iteration index when possible):
  python attach_npy_to_vtk.py --npy path/to/npy_dir --vtk path/to/vtk_dir --output path/to/out_dir
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import meshio
import numpy as np


VTK_EXTS = {".vtk", ".vtu"}


def _extract_index(path: Path) -> int | None:
    """Extract the first integer found in filename stem, if any."""
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else None


def _collect_npy(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix != ".npy":
            raise ValueError(f"Expected a .npy file, got: {path}")
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.npy"))
        if not files:
            raise ValueError(f"No .npy files found in directory: {path}")
        return files
    raise FileNotFoundError(f"Path not found: {path}")


def _collect_vtk(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in VTK_EXTS:
            raise ValueError(f"Expected .vtk or .vtu file, got: {path}")
        return [path]
    if path.is_dir():
        files = []
        for ext in VTK_EXTS:
            files.extend(path.glob(f"*{ext}"))
        files = sorted(files)
        if not files:
            raise ValueError(f"No .vtk/.vtu files found in directory: {path}")
        return files
    raise FileNotFoundError(f"Path not found: {path}")


def _pair_files(npy_files: list[Path], vtk_files: list[Path]) -> list[tuple[Path, Path]]:
    """Pair files by iteration index when available; else by sorted order."""
    npy_by_idx = {idx: p for p in npy_files if (idx := _extract_index(p)) is not None}
    vtk_by_idx = {idx: p for p in vtk_files if (idx := _extract_index(p)) is not None}

    common = sorted(set(npy_by_idx) & set(vtk_by_idx))
    if common:
        return [(npy_by_idx[i], vtk_by_idx[i]) for i in common]

    if len(npy_files) != len(vtk_files):
        raise ValueError(
            "Could not pair by filename index, and file counts differ "
            f"(npy={len(npy_files)}, vtk={len(vtk_files)})."
        )
    return list(zip(sorted(npy_files), sorted(vtk_files)))


def _attach_nodal_solution(npy_path: Path, vtk_path: Path, out_path: Path, field_name: str) -> None:
    mesh = meshio.read(vtk_path)
    u = np.load(npy_path)

    n_points = mesh.points.shape[0]
    u = np.asarray(u).reshape(-1)
    if u.shape[0] != n_points:
        raise ValueError(
            f"Size mismatch for {npy_path.name} and {vtk_path.name}: "
            f"solution has {u.shape[0]} values, mesh has {n_points} points."
        )

    mesh.point_data[field_name] = u
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(out_path, mesh)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npy", required=True, help="Path to .npy file or directory of .npy files")
    parser.add_argument("--vtk", required=True, help="Path to .vtk/.vtu file or directory of mesh files")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output file or directory. If omitted, output is written next to each mesh "
            "as <mesh_stem>_with_u<ext>."
        ),
    )
    parser.add_argument("--field-name", default="u", help="Point-data field name (default: u)")

    args = parser.parse_args()

    npy_path = Path(args.npy).expanduser().resolve()
    vtk_path = Path(args.vtk).expanduser().resolve()
    npy_files = _collect_npy(npy_path)
    vtk_files = _collect_vtk(vtk_path)

    pairs = _pair_files(npy_files, vtk_files)

    output = Path(args.output).expanduser().resolve() if args.output else None
    if output and output.suffix.lower() in VTK_EXTS:
        if len(pairs) != 1:
            raise ValueError("When --output is a file path, exactly one npy/vtk pair is required.")
        out_files = [output]
    elif output:
        output.mkdir(parents=True, exist_ok=True)
        out_files = [output / f"{vtk.stem}_with_u{vtk.suffix}" for _, vtk in pairs]
    else:
        out_files = [vtk.parent / f"{vtk.stem}_with_u{vtk.suffix}" for _, vtk in pairs]

    for (npy_file, vtk_file), out_file in zip(pairs, out_files):
        _attach_nodal_solution(npy_file, vtk_file, out_file, args.field_name)
        print(f"Attached {npy_file.name} -> {vtk_file.name} -> {out_file}")


if __name__ == "__main__":
    main()
