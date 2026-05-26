import logging
import meshio
logging.getLogger("meshio").setLevel(logging.ERROR)
import gmsh
import dolfinx
from dolfinx.io import gmsh as gmshio
from dolfinx.mesh import exterior_facet_indices
import numpy as np
import basix.ufl
from mpi4py import MPI
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

# Medit keywords that meshio either warns about or fails hard on.
# We strip these sections before handing the file to meshio, using a
# temporary copy so the original mesh file stays intact for subsequent
# MMG / ParMmg calls that may need these sections.
_MEDIT_MESHIO_UNSUPPORTED = frozenset({
    "RequiredEdges",
    "RequiredVertices",
    "RequiredTriangles",
    "RequiredTetrahedra",
    "Ridges",
    "Corners",
    "NormalsAtVertices",
    "Tangents",
    "TangentAtVertices",
})


def _medit_strip_unsupported(path: Path) -> str:
    """Return the content of a Medit .mesh file with unsupported sections removed."""
    lines = Path(path).read_text().splitlines()
    out = []
    i = 0
    while i < len(lines):
        keyword = lines[i].strip()
        if keyword in _MEDIT_MESHIO_UNSUPPORTED:
            i += 1  # skip keyword line
            # skip optional blank lines before the count
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines):
                try:
                    count = int(lines[i].strip())
                    i += 1  # skip count line
                    skipped = 0
                    while i < len(lines) and skipped < count:
                        if lines[i].strip():
                            skipped += 1
                        i += 1
                except ValueError:
                    pass  # no integer count line — just skip the keyword itself
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out) + "\n"

# def read_medit_to_dolfinx(path: str) -> dolfinx.mesh.Mesh:
#     gmsh.initialize()
#     gmsh.option.setNumber("General.Verbosity", 0)
#     gmsh.open(path)

#     # MMG .mesh files have no physical groups, but model_to_mesh needs them.
#     # Tag all 3D entities as a single physical group.
#     volumes = gmsh.model.getEntities(dim=3)
#     if volumes:
#         vol_tags = [v[1] for v in volumes]
#         gmsh.model.addPhysicalGroup(3, vol_tags, tag=1)

#     mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, rank=0, gdim=3)
#     gmsh.finalize()
#     return mesh_data.mesh

def read_medit_to_dolfinx(path: str) -> dolfinx.mesh.Mesh:
    clean = _medit_strip_unsupported(Path(path))
    with tempfile.NamedTemporaryFile(suffix=".mesh", mode="w", delete=False) as tmp:
        tmp.write(clean)
        tmp_path = tmp.name
    try:
        m = meshio.read(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    tets = m.cells_dict.get("tetra")
    if tets is None:
        raise ValueError("No tetrahedra found in the mesh file.")

    # Force contiguous arrays with correct dtypes
    points = np.ascontiguousarray(m.points[:, :3], dtype=np.float64)
    cells  = np.ascontiguousarray(tets, dtype=np.int64)

    coord_element = basix.ufl.element(
        "Lagrange", "tetrahedron", 1, shape=(3,)
    )
    msh = dolfinx.mesh.create_mesh(
        MPI.COMM_SELF, cells, coord_element, points
    )
    return msh

def upgrade_mesh_to_p2(msh_p1: dolfinx.mesh.Mesh) -> dolfinx.mesh.Mesh:
    """Upgrade a P1 tetrahedral mesh to P2 geometry by generating edge midpoint nodes.

    The topology (cells and vertices) is unchanged; DOLFINx interpolates the
    identity map into the P2 coordinate space to produce midpoint coordinates.
    """
    from dolfinx import fem
    p2_element = basix.ufl.element("Lagrange", "tetrahedron", 2, shape=(3,))
    V = fem.functionspace(msh_p1, p2_element)
    xfn = fem.Function(V)
    xfn.interpolate(lambda x: x)
    n_cells = msh_p1.topology.index_map(3).size_local
    cell_dofs = np.array([V.dofmap.cell_dofs(c) for c in range(n_cells)], dtype=np.int64)
    coords = xfn.x.array.reshape(-1, 3)
    return dolfinx.mesh.create_mesh(MPI.COMM_SELF, cell_dofs, p2_element, coords)


def build_dolfinx_to_medit_map(msh):
    """Return perm such that M_reordered[perm[i]] = M_dolfinx[i].
    Uses the built-in input_global_indices provided by DOLFINx."""
    return msh.geometry.input_global_indices

def write_metric_sol(mesh_path: Path | str, M_all: np.ndarray, sol_path: Path | str | None = None,) -> Path:

    mesh_path = Path(mesh_path)

    if sol_path is None:
        sol_path = mesh_path.with_suffix(".sol")
    sol_path = Path(sol_path)

    if M_all.ndim != 3 or M_all.shape[1] != 3 or M_all.shape[2] != 3:
        raise ValueError(
            f"M_all must have shape (N, 3, 3), got {M_all.shape}"
        )
    N = M_all.shape[0]

    n_mesh = _read_vertex_count(mesh_path)
    if N != n_mesh:
        raise ValueError(
            f"M_all has {N} entries but {mesh_path.name} has {n_mesh} vertices"
        )

    with open(sol_path, "w") as sf:
        # Header — must match the companion .mesh version
        sf.write("MeshVersionFormatted 2\n")
        sf.write("Dimension 3\n\n")

        sf.write(f"SolAtVertices\n{N}\n")
        sf.write("1 3\n\n")

        for M in M_all:
            sf.write(
                f"{M[0, 0]:.14e}  {M[0, 1]:.14e}  {M[1, 1]:.14e} "
                f"{M[0, 2]:.14e}  {M[1, 2]:.14e}  "
                f"{M[2, 2]:.14e}\n"
            )

        sf.write("\nEnd\n")

    print(f"[sol] wrote {N} tensors → {sol_path.name}")
    return sol_path


def _read_vertex_count(mesh_path: Path) -> int:

    with open(mesh_path, "r") as mf:
        for line in mf:
            if line.strip() == "Vertices":
                count_line = next(mf).strip()
                return int(count_line)
    raise ValueError(f"No 'Vertices' section found in {mesh_path}")

def _signed_tet_volume(pts, i1, i2, i3, i4):
    v1 = pts[i2-1] - pts[i1-1]
    v2 = pts[i3-1] - pts[i1-1]
    v3 = pts[i4-1] - pts[i1-1]
    return float(np.linalg.det(np.column_stack([v1, v2, v3])))

def write_cube_mesh(path):

    verts = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [-1, 1, -1],
        [1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, 1]
    ])

    # verts = np.array([
    # [0,0,0], [1,0,0], [0,1,0], [1,1,0],
    # [0,0,1], [1,0,1], [0,1,1], [1,1,1],
    # ])

    initial_tets = [
        [1, 2, 4, 8],
        [1, 4, 3, 8],
        [1, 3, 7, 8],
        [1, 7, 5, 8],
        [1, 5, 6, 8],
        [1, 6, 2, 8]
    ]

    tets = []
    for (i1, i2, i3, i4) in initial_tets:
        if _signed_tet_volume(verts, i1, i2, i3, i4) > 0:
            tets.append([i1, i2, i3, i4])
        else:
            tets.append([i1, i3, i2, i4])
        
    triangles = [
        (1, 2, 4, 1), (1, 4, 3, 1),
        (5, 8, 6, 2), (5, 7, 8, 2),
        (1, 6, 2, 3), (1, 5, 6, 3),
        (3, 4, 8, 4), (3, 8, 7, 4),
        (1, 3, 7, 5), (1, 7, 5, 5),
        (2, 6, 8, 6), (2, 8, 4, 6)
    ]

    with open(path, 'w') as mf:
        mf.write("MeshVersionFormatted 2\n")
        mf.write("Dimension 3\n\n")

        mf.write("Vertices\n")
        mf.write(f"{len(verts)}\n")
        for (x, y, z) in verts:
            mf.write(f"{x:.6f} {y:.6f} {z:.6f} 0\n")
        
        mf.write("\nTriangles\n")
        mf.write(f"{len(triangles)}\n")
        for (i1, i2, i3, tag) in triangles:
            mf.write(f"{i1} {i2} {i3} {tag}\n")

        mf.write("\nTetrahedra\n")
        mf.write(f"{len(tets)}\n")
        for (a, b, c, d) in tets:
            mf.write(f"{a} {b} {c} {d} 1\n")

        mf.write("\nEnd\n")

def run_mmg3d(mmg_exe, args):
    cmd = [mmg_exe] + [str(arg) for arg in args]
    print(f"[MMG3D] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"MMG3D failed with return code {result.returncode}")

def generate_initial_mesh(coarse_path, output_path, mmg_exe, hmax=0.1, hgrad=-1):
    run_mmg3d(mmg_exe, [
        "-in",    coarse_path,
        "-out",   output_path,
        "-hmax",  hmax,
        "-hgrad", hgrad
    ])
    nv = _read_vertex_count(output_path)
    print(f"Generated initial mesh with {nv} vertices.")
    return output_path

def adapt_mesh(input_path, output_path, M, hgrad, hmin, hmax, mmg_exe):

    sol_path = output_path.replace('.mesh', '.sol')

    write_metric_sol(input_path, M, sol_path)

    run_mmg3d(mmg_exe, [
        "-in",    input_path,
        "-sol",   sol_path,
        "-out",   output_path,
        "-hgrad", hgrad,
        "-hmin",  hmin,
        "-hmax",  hmax
    ])

    nv_in  = _read_vertex_count(input_path)
    nv_out = _read_vertex_count(output_path)
    print(f"Adapted mesh from {nv_in} to {nv_out} vertices.")

    return output_path


def build_metric(
    h_p: np.ndarray,
    Q: np.ndarray,
    perm: np.ndarray,
    mesh_path: Path | str,
    sol_path: Path | str,
) -> Path:

    # n_dolfinx: vertices DOLFINx knows about (h_p / Q are indexed by this)
    # n_medit:   vertices stored in the MEDIT file (M_reordered must match this)
    # MMG3D can produce required/corner vertices that sit in the Vertices section
    # but are unreferenced by any tetrahedron; DOLFINx drops those but
    # input_global_indices still indexes into the full original vertex list.
    n_dolfinx = h_p.shape[0]
    n_medit = _read_vertex_count(mesh_path)

    M_all = np.zeros((n_dolfinx, 3, 3))
    for P in range(n_dolfinx):
        D = np.diag(1.0 / h_p[P] ** 2)
        Q_P = Q[P]
        M_all[P] = Q_P @ D @ Q_P.T

    M_reordered = np.zeros((n_medit, 3, 3))
    for i in range(n_dolfinx):
        M_reordered[perm[i]] = M_all[i]

    return write_metric_sol(mesh_path, M_reordered, sol_path)


def adapt_mesh_mmg(
    input_path: Path | str,
    output_path: Path | str,
    sol_path: Path | str,
    mmg_log_file: Path | str,
    mmg_exe: str,
    hgrad: float,
    hmin: float,
    hmax: float,
    vtk_dir: Path | str | None = None,
    extra_args: list | None = None,
) -> tuple[str, str | None]:

    cmd = [
        mmg_exe,
        "-in", str(input_path),
        "-sol", str(sol_path),
        "-out", str(output_path),
        "-hgrad", str(hgrad),
        "-hmin", str(hmin),
        "-hmax", str(hmax),
    ]
    if extra_args:
        cmd.extend(extra_args)
    print(f"  Command: {' '.join(cmd)}")

    with open(mmg_log_file, "a") as log_f:
        log_f.write(f"\n{'=' * 70}\n")
        log_f.write(f"ADAPT {Path(input_path).name} -> {Path(output_path).name}\n")
        log_f.write(f"{'=' * 70}\n")
        result = subprocess.run(
            cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"MMG3D failed with return code {result.returncode}. "
            f"Check log: {mmg_log_file}"
        )
    print("  MMG3D completed successfully")

    vtu_path = to_vtu(str(output_path), output_dir=vtk_dir)
    if vtu_path:
        print(f"  VTU saved: {Path(vtu_path).name}")

    return str(output_path), vtu_path


def adapt_mesh_parmmg(
    input_path: "Path | str",
    output_path: "Path | str",
    sol_path: "Path | str",
    parmmg_log_file: "Path | str",
    parmmg_exe: str,
    hgrad: float,
    hmin: float,
    hmax: float,
    np_mpi: int = 4,
    niter: int = 6,
    nlayers: int = 3,
    mesh_size: int | None = None,
    mpirun_exe: str = "mpirun",
    vtk_dir: "Path | str | None" = None,
    extra_args: list | None = None,
) -> tuple[str, str | None]:
    """Run ParMmg for metric-based anisotropic adaptation.

    Drop-in replacement for ``adapt_mesh_mmg``.  Pass ``-hgradreq`` via
    ``extra_args``, e.g. ``extra_args=["-hgradreq", "3.0"]``.

    Parameters
    ----------
    np_mpi : Number of MPI processes.
    niter : Remeshing-repartitioning iterations inside ParMmg (``-niter``).
    nlayers : Interface displacement layers per repartitioning (``-nlayers``).
    mesh_size : Target elements per sequential Mmg chunk (``-mesh-size``); None → auto.
    """
    cmd = [
        mpirun_exe,
        "-np", str(np_mpi),
        parmmg_exe,
        "-in",      str(input_path),
        "-sol",     str(sol_path),
        "-out",     str(output_path),
        "-hgrad",   str(hgrad),
        "-hmin",    str(hmin),
        "-hmax",    str(hmax),
        "-niter",   str(niter),
        "-nlayers", str(nlayers),
    ]
    if mesh_size is not None:
        cmd.extend(["-mesh-size", str(mesh_size)])
    if extra_args:
        cmd.extend([str(a) for a in extra_args])

    print(f"  Command: {' '.join(cmd)}")

    with open(parmmg_log_file, "a") as log_f:
        log_f.write(f"\n{'=' * 70}\n")
        log_f.write(
            f"PARMMG ADAPT {Path(input_path).name} -> "
            f"{Path(output_path).name}  ({np_mpi} procs)\n"
        )
        log_f.write(f"{'=' * 70}\n")
        result = subprocess.run(
            cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"ParMmg failed with return code {result.returncode}. "
            f"Check log: {parmmg_log_file}"
        )
    print("  ParMmg completed successfully")

    vtu_path = to_vtu(str(output_path), output_dir=vtk_dir)
    if vtu_path:
        print(f"  VTU saved: {Path(vtu_path).name}")

    return str(output_path), vtu_path


def save_computed_quantities(
    compute_errs_dir: Path | str,
    u_h,
    grad_uh,
    eta_k: np.ndarray,
    eta_zz_fn,
    time: float = 0.0,
) -> None:
    """Persist solution quantities to an ADIOS2 BP4 checkpoint.

    Parameters
    ----------
    compute_errs_dir :
        Directory where ``computed_quantities.bp`` will be written.
    u_h :
        FEM solution Function.
    grad_uh :
        DG0 vector Function holding ``grad(u_h)``.
    eta_k :
        1-D numpy array of cell-wise anisotropic error indicator values.
    eta_zz_fn :
        DG0 vector Function for the ZZ gradient error ``grad(u_h) - Pi_h(grad(u_h))``.
    time :
        Pseudo-time stamp stored in the checkpoint (default 0.0).
    """
    import adios4dolfinx
    import adios2
    from dolfinx import fem as _fem

    out_dir = Path(compute_errs_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    mesh = u_h.function_space.mesh

    # Wrap the raw numpy array in a DG0 scalar Function for storage
    V_eta = _fem.functionspace(mesh, ("DG", 0))
    eta_k_fn = _fem.Function(V_eta, name="eta_k")
    eta_k_fn.x.array[:] = eta_k

    filename = out_dir / "computed_quantities.bp"

    # Write mesh first (creates / overwrites the BP4 file)
    adios4dolfinx.write_mesh(filename, mesh, mode=adios2.Mode.Write)
    # Append each function to the same file
    adios4dolfinx.write_function(filename, u_h,        mode=adios2.Mode.Append, time=time, name="u")
    adios4dolfinx.write_function(filename, grad_uh,    mode=adios2.Mode.Append, time=time, name="grad_u")
    adios4dolfinx.write_function(filename, eta_k_fn,   mode=adios2.Mode.Append, time=time, name="eta_k")
    adios4dolfinx.write_function(filename, eta_zz_fn,  mode=adios2.Mode.Append, time=time, name="eta_zz")

    print(f"  Saved computed quantities → {filename}")

def to_vtu(mesh_path, output_dir=None, write_solution=False, u_h=None, dof_to_medit=None):
    """Convert a Medit .mesh file to VTU.

    If ``output_dir`` is given, the VTU is placed there instead of next to the
    source mesh.

    When ``write_solution`` is True, ``u_h`` is also stored in VTU point data
    under the name ``u_h``.
    """
    try:
        import meshio
    except ImportError:
        print("[WARNING] meshio not installed.  Run: pip install meshio")
        print("          Or open .mesh files directly in ParaView ≥ 5.10")
        return None

    if output_dir is not None:
        vtu_path = str(Path(output_dir) / Path(mesh_path).with_suffix(".vtu").name)
    else:
        vtu_path = mesh_path.replace(".mesh", ".vtu")

    clean = _medit_strip_unsupported(Path(mesh_path))
    with tempfile.NamedTemporaryFile(suffix=".mesh", mode="w", delete=False) as tmp:
        tmp.write(clean)
        tmp_path = tmp.name
    try:
        mesh = meshio.read(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Keep only tetrahedral cells (discard any lower-dim leftovers)
    tet_cells = [c for c in mesh.cells if c.type == "tetra"]
    if not tet_cells:
        tet_cells = mesh.cells

    point_data = {}
    if write_solution:
        if u_h is None:
            raise ValueError("write_solution=True requires u_h to be provided")

        if hasattr(u_h, "x") and hasattr(u_h.x, "array"):
            values = np.asarray(u_h.x.array, dtype=np.float64).reshape(-1)
        else:
            values = np.asarray(u_h, dtype=np.float64).reshape(-1)

        n_points = mesh.points.shape[0]

        if dof_to_medit is not None:
            perm = np.asarray(dof_to_medit, dtype=np.int64)
            n_dof = perm.size
            # MMG3D can produce required/corner vertices that are stored in the
            # Medit file but unreferenced by any tetrahedron; DOLFINx drops those,
            # so n_dof may be smaller than n_points by one (or more).
            if values.size == n_dof:
                src = values
            elif values.size % n_dof == 0:
                src = values.reshape(n_dof, values.size // n_dof)
            else:
                raise ValueError(
                    f"u_h size {values.size} is not a multiple of dof_to_medit "
                    f"size {n_dof}"
                )
            if src.ndim == 1:
                reordered = np.zeros(n_points, dtype=np.float64)
            else:
                reordered = np.zeros((n_points, src.shape[1]), dtype=np.float64)
            reordered[perm] = src
            point_values = reordered
        else:
            if values.size == n_points:
                point_values = values
            elif values.size % n_points == 0:
                n_comp = values.size // n_points
                point_values = values.reshape(n_points, n_comp)
            else:
                raise ValueError(
                    "u_h size is incompatible with mesh points: "
                    f"got {values.size} values for {n_points} points"
                )

        point_data["u_h"] = point_values

    meshio.write(vtu_path, meshio.Mesh(points=mesh.points, cells=tet_cells, point_data=point_data))
    print(f"[VTK]  {mesh_path}  to  {vtu_path}")
    return vtu_path

def get_boundary_vertex_indices(msh) -> np.ndarray:
    """Return 0-based geometry vertex indices of all boundary vertices."""
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    ext_f = exterior_facet_indices(msh.topology)
    f2v = msh.topology.connectivity(fdim, 0)
    bverts = set()
    for fi in ext_f:
        bverts.update(f2v.links(fi).tolist())
    return np.fromiter(bverts, dtype=np.intp)


def snap_tokamak_wall_vertices(
    mesh_path: Path | str,
    r_inner: float,
    r_outer: float,
    z_bottom: float | None = 0,
    z_top: float | None = 800,
    normal_z_thresh: float = 0.5,
) -> dict:
    """Snap boundary vertices onto exact tokamak geometry, with tet-safety.

    Cylindrical-wall vertices are projected to r_inner or r_outer.
    Cap vertices (if z_bottom/z_top given) are projected to the exact z-plane.
    A bisection fallback prevents any tetrahedron from being inverted.

    Returns
    -------
    dict with keys "inner", "outer", "bottom", "top", each mapping to:
        total, snapped, bisected, mean_dist, std_dist, max_dist
    (dist fields measure residual distance to exact surface for bisected vertices only)
    """
    mesh_path = Path(mesh_path)
    lines = mesh_path.read_text().splitlines()

    def _find_section(name: str) -> int:
        for idx, ln in enumerate(lines):
            if ln.strip() == name:
                return idx
        return -1

    def _skip_blank(idx: int) -> int:
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        return idx

    # ---- Parse Vertices ----
    vi = _find_section("Vertices")
    if vi < 0:
        raise ValueError("No 'Vertices' section")
    vi = _skip_blank(vi + 1)
    n_verts = int(lines[vi].strip())
    vert_start = vi + 1

    coords = np.empty((n_verts, 3))
    refs: list[str] = []
    for j in range(n_verts):
        parts = lines[vert_start + j].split()
        coords[j] = [float(parts[0]), float(parts[1]), float(parts[2])]
        refs.append(parts[3])

    # ---- Parse Triangles → classify wall vs cap vertices ----
    wall_verts: set[int] = set()
    cap_verts: set[int] = set()

    ti = _find_section("Triangles")
    if ti >= 0:
        ti = _skip_blank(ti + 1)
        n_tri = int(lines[ti].strip())
        for k in range(n_tri):
            parts = lines[ti + 1 + k].split()
            i0, i1, i2 = int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1
            a = coords[i1] - coords[i0]
            b = coords[i2] - coords[i0]
            nx = a[1]*b[2] - a[2]*b[1]
            ny = a[2]*b[0] - a[0]*b[2]
            nz = a[0]*b[1] - a[1]*b[0]
            n_mag = np.sqrt(nx*nx + ny*ny + nz*nz)
            if n_mag < 1e-14:
                continue
            if abs(nz) / n_mag < normal_z_thresh:
                wall_verts.update([i0, i1, i2])
            else:
                cap_verts.update([i0, i1, i2])

    # ---- Parse Tetrahedra (needed for inversion check) ----
    ei = _find_section("Tetrahedra")
    if ei < 0:
        raise ValueError("No 'Tetrahedra' section")
    ei = _skip_blank(ei + 1)
    n_tets = int(lines[ei].strip())
    tets = np.empty((n_tets, 4), dtype=np.int64)
    for k in range(n_tets):
        parts = lines[ei + 1 + k].split()
        tets[k] = [int(parts[m]) - 1 for m in range(4)]

    # Build vertex → tet adjacency
    vert_to_tets: dict[int, list[int]] = {j: [] for j in (wall_verts | cap_verts)}
    for k in range(n_tets):
        for v in tets[k]:
            if v in vert_to_tets:
                vert_to_tets[v].append(k)

    def _signed_vol(c, tet_idx):
        v0, v1, v2, v3 = tets[tet_idx]
        d1 = c[v1] - c[v0]
        d2 = c[v2] - c[v0]
        d3 = c[v3] - c[v0]
        return np.dot(d1, np.cross(d2, d3))

    bbox_diag = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))
    vol_threshold = 1e-6 * (bbox_diag / n_verts**(1/3))**3

    def _all_tets_ok(c, vert_idx):
        return all(_signed_vol(c, t) > vol_threshold for t in vert_to_tets[vert_idx])

    # ---- Assign each vertex to a face and compute target positions ----
    Rxy = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
    coords_new = coords.copy()
    vert_face: dict[int, str] = {}

    for j in wall_verts:
        if Rxy[j] < 1e-14:
            continue
        face = "inner" if abs(Rxy[j] - r_inner) < abs(Rxy[j] - r_outer) else "outer"
        vert_face[j] = face
        R_target = r_inner if face == "inner" else r_outer
        scale = R_target / Rxy[j]
        coords_new[j, 0] = coords[j, 0] * scale
        coords_new[j, 1] = coords[j, 1] * scale

    if z_bottom is not None or z_top is not None:
        for j in cap_verts:
            if j in wall_verts:
                continue
            z_cur = coords[j, 2]
            if z_bottom is not None and z_top is not None:
                face = "bottom" if abs(z_cur - z_bottom) < abs(z_cur - z_top) else "top"
            elif z_bottom is not None:
                face = "bottom"
            else:
                face = "top"
            vert_face[j] = face
            coords_new[j, 2] = z_bottom if face == "bottom" else z_top

    # ---- Per-face statistics ----
    face_stats: dict[str, dict] = {
        f: {"total": 0, "snapped": 0, "bisected": 0, "bisected_dists": []}
        for f in ("inner", "outer", "bottom", "top")
    }
    for j, face in vert_face.items():
        face_stats[face]["total"] += 1

    # ---- Apply with bisection safety ----
    n_snapped = 0
    n_bisected = 0
    for j in (wall_verts | cap_verts):
        if np.allclose(coords[j], coords_new[j], atol=1e-14):
            continue

        face = vert_face.get(j)
        old = coords[j].copy()
        coords[j] = coords_new[j]

        if _all_tets_ok(coords, j):
            n_snapped += 1
            if face:
                face_stats[face]["snapped"] += 1
        else:
            disp = coords_new[j] - old
            alpha_lo, alpha_hi = 0.0, 1.0
            for _ in range(30):
                alpha_mid = 0.5 * (alpha_lo + alpha_hi)
                coords[j] = old + alpha_mid * disp
                if _all_tets_ok(coords, j):
                    alpha_lo = alpha_mid
                else:
                    alpha_hi = alpha_mid
            coords[j] = old + alpha_lo * disp
            n_snapped += 1
            n_bisected += 1
            if face:
                face_stats[face]["snapped"] += 1
                face_stats[face]["bisected"] += 1
                if face == "inner":
                    dist = abs(np.sqrt(coords[j, 0]**2 + coords[j, 1]**2) - r_inner)
                elif face == "outer":
                    dist = abs(np.sqrt(coords[j, 0]**2 + coords[j, 1]**2) - r_outer)
                elif face == "bottom":
                    dist = abs(coords[j, 2] - z_bottom)
                else:
                    dist = abs(coords[j, 2] - z_top)
                face_stats[face]["bisected_dists"].append(dist)

    # ---- Write back ----
    for j in range(n_verts):
        lines[vert_start + j] = (
            f"{coords[j, 0]:.14e} {coords[j, 1]:.14e} "
            f"{coords[j, 2]:.14e} {refs[j]}"
        )
    mesh_path.write_text("\n".join(lines) + "\n")

    # ---- Build result dict and print summary ----
    result: dict[str, dict] = {}
    for face, stats in face_stats.items():
        dists = stats["bisected_dists"]
        result[face] = {
            "total":     stats["total"],
            "snapped":   stats["snapped"],
            "bisected":  stats["bisected"],
            "mean_dist": float(np.mean(dists)) if dists else 0.0,
            "std_dist":  float(np.std(dists))  if dists else 0.0,
            "max_dist":  float(np.max(dists))  if dists else 0.0,
        }

    print(
        f"[snap] {mesh_path.name}: snapped {n_snapped} vertices "
        f"({n_bisected} needed bisection fallback)"
    )
    for face, s in result.items():
        dist_str = (
            f"  dist: mean={s['mean_dist']:.3e}  std={s['std_dist']:.3e}  max={s['max_dist']:.3e}"
            if s["bisected"] > 0 else ""
        )
        print(
            f"  [{face:6s}] total={s['total']:5d}  snapped={s['snapped']:5d}"
            f"  bisected={s['bisected']:5d}{dist_str}"
        )
    return result

def write_dolfinx_to_medit(msh, path: str) -> None:
    tdim = 3
    fdim = 2

    msh.topology.create_connectivity(fdim, 0)
    msh.topology.create_connectivity(fdim, tdim)

    coords = msh.geometry.x   # (N, 3) float64
    gdmap  = np.asarray(msh.geometry.dofmap)  # (K, 4) 0-based
    N = coords.shape[0]
    K = gdmap.shape[0]

    ext_f = exterior_facet_indices(msh.topology)  # 1D array of facet indices

    f2v = msh.topology.connectivity(fdim, 0)

    with open(path, "w") as mf:
        mf.write("MeshVersionFormatted 2\nDimension 3\n\n")

        # Vertices: row k (1-based in file) = coords[k-1] (0-based in array)
        mf.write(f"Vertices\n{N}\n")
        for i in range(N):
            # ref = 0 means "no special attribute"
            mf.write(f"{coords[i,0]:.14e} {coords[i,1]:.14e} {coords[i,2]:.14e} 0\n")

        # Triangles: boundary faces, ref = 1 (any nonzero integer works)
        mf.write(f"\nTriangles\n{len(ext_f)}\n")
        for fi in ext_f:
            verts = f2v.links(fi)
            mf.write(f"{verts[0]+1} {verts[1]+1} {verts[2]+1} 1\n")

        # Tetrahedra: gdmap is already 0-based geometry indices
        mf.write(f"\nTetrahedra\n{K}\n")
        for c in range(K):
            v = gdmap[c]
            mf.write(f"{v[0]+1} {v[1]+1} {v[2]+1} {v[3]+1} 1\n")

        mf.write("\nEnd\n")
