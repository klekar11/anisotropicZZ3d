import meshio
import gmsh
import dolfinx
from dolfinx.io import gmsh as gmshio
from dolfinx.mesh import exterior_facet_indices
import numpy as np
import basix.ufl
from mpi4py import MPI
import subprocess
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

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
    m = meshio.read(path)
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
) -> tuple[str, str | None]:

    cmd = [
        mmg_exe,
        "-in", str(input_path),
        "-sol", str(sol_path),
        "-out", str(output_path),
        "-hgrad", str(hgrad),
        "-hmin", str(hmin),
        "-hmax", str(hmax)
    ]
    print(f"  Command: {' '.join(cmd)}")

    with open(mmg_log_file, "a") as log_f:
        log_f.write(f"\n{'=' * 70}\n")
        log_f.write(f"ADAPT {Path(input_path).name} -> {Path(output_path).name}\n")
        log_f.write(f"{'=' * 70}\n")
        result = subprocess.run(
            cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True
        )

    if result.returncode != 0:
        print(f"  WARNING: MMG3D returned code {result.returncode}")
    else:
        print("  MMG3D completed successfully")

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

    mesh = meshio.read(mesh_path)

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

        if dof_to_medit is not None:
            perm = np.asarray(dof_to_medit, dtype=np.int64)
            if perm.size != n_points:
                raise ValueError(
                    "dof_to_medit size mismatch: "
                    f"got {perm.size}, expected {n_points}"
                )
            reordered = np.empty_like(point_values)
            reordered[perm] = point_values
            point_values = reordered

        point_data["u_h"] = point_values

    meshio.write(vtu_path, meshio.Mesh(points=mesh.points, cells=tet_cells, point_data=point_data))
    print(f"[VTK]  {mesh_path}  to  {vtu_path}")
    return vtu_path

# a bit overkill maybe just better to generate the first mesh in mmg
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