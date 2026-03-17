import os
import numpy as np
import subprocess

MMG3D_EXE='mmg3d_O3'
OUTPUT_DIR='cube_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

H_X = 0.1
H_Y = 0.1
H_Z = 0.01

LAM_X = 1 / H_X**2
LAM_Y = 1 / H_Y**2
LAM_Z = 1 / H_Z**2

H_MIN = 10**(-5)
H_MAX = 0.4 

METRIC_3D = np.diag([LAM_X, LAM_Y, LAM_Z])

def _signed_tet_volume(pts, i1, i2, i3, i4):
    v1 = pts[i2-1] - pts[i1-1]
    v2 = pts[i3-1] - pts[i1-1]
    v3 = pts[i4-1] - pts[i1-1]
    return float(np.linalg.det(np.column_stack([v1, v2, v3])))

def write_cube_mesh(path):

    # verts = np.array([
    #     [-1, -1, -1],
    #     [1, -1, -1],
    #     [-1, 1, -1],
    #     [1, 1, -1],
    #     [-1, -1, 1],
    #     [1, -1, 1],
    #     [-1, 1, 1],
    #     [1, 1, 1]
    # ])

    verts = np.array([
    [0,0,0], [1,0,0], [0,1,0], [1,1,0],
    [0,0,1], [1,0,1], [0,1,1], [1,1,1],
    ])

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

def read_corner_and_ridge_vertex_indices(mesh_path):

    special = set()
    edges_by_index = {}
    ridge_edge_indices = set()
    with open(mesh_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        token = lines[i].strip()

        # Corners section: each line is a single vertex index
        if token == "Corners":
            n = int(lines[i+1].strip())
            for j in range(n):
                special.add(int(lines[i+2+j].strip()))
            i += 2 + n

        elif token == "Edges":
            n = int(lines[i+1].strip())
            for j in range(n):
                parts = lines[i+2+j].split()
                # edge index is 1-based (j+1)
                edges_by_index[j+1] = (int(parts[0]), int(parts[1]))
            i += 2 + n
        
        elif token == "Ridges":
            n = int(lines[i+1].strip())
            for j in range(n):
                ridge_edge_indices.add(int(lines[i+2+j].strip()))
            i += 2 + n

        else:
            i += 1
        
        for edge_idx in ridge_edge_indices:
            if edge_idx in edges_by_index:
                v1, v2 = edges_by_index[edge_idx]
                special.add(v1)
                special.add(v2)
            else:
                print(f"[WARN] Ridge edge index {edge_idx} not found in Edges section")

    return special


def read_boundary_vertex_indices(mesh_path):
    
    boundary = set()
    with open(mesh_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        token = lines[i].strip()
        if token == "Triangles":
            n = int(lines[i+1].strip())
            for j in range(n):
                parts = lines[i+2+j].split()
                boundary.add(int(parts[0]))
                boundary.add(int(parts[1]))
                boundary.add(int(parts[2]))
                # parts[3] is the face tag — not a vertex index
            i += 2 + n
        else:
            i += 1

    return boundary


def read_num_vertices(path):
    with open(path, 'r') as mf:
        lines = mf.readlines()
        for i, line in enumerate(lines):
            if line.strip() == "Vertices":
                return int(lines[i+1].strip())
    raise ValueError("Could not find 'Vertices' section in the mesh file.")

def write_sol_file(path, mesh_path, M):
    
    nv = read_num_vertices(mesh_path)

    # # Read corner vertex indices (1-based)
    # corners = read_corner_and_ridge_vertex_indices(mesh_path)
    # # Diagnostic: how many vertices get each metric
    # print(f"[SOL] Special vertices (corners + ridges): {len(corners)} / {nv}")
    # print(f"      These get isotropic h = {1/min(M[0,0],M[1,1],M[2,2])**0.5:.4f}")
    # print(f"      Interior gets anisotropic h = ({1/M[0,0]**0.5:.3f}, "
    #       f"{1/M[1,1]**0.5:.3f}, {1/M[2,2]**0.5:.3f})")

    boundary = read_boundary_vertex_indices(mesh_path)
    print(f"[SOL] Boundary vertices (all faces): {len(boundary)} / {nv}")

    # Full anisotropic metric components
    m11, m12, m13 = float(M[0,0]), float(M[0,1]), float(M[1,1])
    m22, m23      = float(M[0,2]), float(M[1,2])
    m33           = float(M[2,2])
    aniso_line = (f"{m11:.10e} {m12:.10e} {m13:.10e} "
                  f"{m22:.10e} {m23:.10e} {m33:.10e}\n")

    # lam_safe = min(m11, m22, m33)   # smallest eigenvalue = largest h
    # iso_line = (f"{lam_safe:.10e} 0.0000000000e+00 0.0000000000e+00 "
    #             f"{lam_safe:.10e} 0.0000000000e+00 {lam_safe:.10e}\n")

    with open(path, "w") as sf:
        sf.write("MeshVersionFormatted 2\nDimension 3\n\n")
        sf.write(f"SolAtVertices\n{nv}\n1 3\n")
        for _ in range(nv):
            sf.write(aniso_line)           # full metric everywhere (no corners)
        # for idx in range(1, nv + 1):           # 1-based vertex index
            # if idx in boundary:
            #     sf.write(iso_line)             # isotropic at corners
            # else:
            #     sf.write(aniso_line)           # full metric everywhere else

        sf.write("End\n")


    print(f"[SOL] Wrote {path}  ({nv} vertices, "
          f"{len(boundary)} isotropic corners, "
          f"{nv - len(boundary)} anisotropic interior)")


def run_mmg3d(args):
    cmd = [MMG3D_EXE] + [str(arg) for arg in args]
    print(f"[MMG3D] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"MMG3D failed with return code {result.returncode}")

def generate_initial_mesh(coarse_path, output_path, hmax=0.1):
    run_mmg3d([
        "-in",    coarse_path,
        "-out",   output_path,
        "-hmax",  hmax,
        "-hgrad", -1
    ])
    nv = read_num_vertices(output_path)
    print(f"Generated initial mesh with {nv} vertices.")
    return output_path

def adapt_mesh(input_path, output_path, M):

    sol_path = output_path.replace('.mesh', '.sol')

    write_sol_file(sol_path, input_path, M)

    run_mmg3d([
        "-in",    input_path,
        "-sol",   sol_path,
        "-out",   output_path,
        "-hgrad", -1
    ])

    nv_in  = read_num_vertices(input_path)
    nv_out = read_num_vertices(output_path)
    print(f"Adapted mesh from {nv_in} to {nv_out} vertices.")

    return output_path

def to_vtu(mesh_path):
    try:
        import meshio
    except ImportError:
        print("[WARNING] meshio not installed.  Run: pip install meshio")
        print("          Or open .mesh files directly in ParaView ≥ 5.10")
        return None

    vtu_path = mesh_path.replace(".mesh", ".vtu")
    mesh = meshio.read(mesh_path)

    # Keep only tetrahedral cells (discard any lower-dim leftovers)
    tet_cells = [c for c in mesh.cells if c.type == "tetra"]
    if not tet_cells:
        tet_cells = mesh.cells   # fallback: keep whatever is there

    meshio.write(
        vtu_path,
        meshio.Mesh(points=mesh.points, cells=tet_cells),
    )
    print(f"[VTK]  {mesh_path}  to  {vtu_path}")
    return vtu_path

def main():
    p = lambda name: os.path.join(OUTPUT_DIR, name)
    print("PHASE 1 — Uniform initial mesh (no metric)")
    coarse_path  = p("cube_coarse.mesh")
    initial_path = p("cube_initial.mesh")

    write_cube_mesh(coarse_path)
    generate_initial_mesh(coarse_path, initial_path, hmax=0.1)
    to_vtu(initial_path)

    print("PHASE 2 — Anisotropic adaptation")
    

    STAGES = [
    #np.diag([1/H_X**2, 1/H_Y**2, 1/0.08**2]), # step 1: h_z = 0.08
    #np.diag([1/H_X**2, 1/H_Y**2, 1/0.04**2]),  # step 2: h_z = 0.04
    #np.diag([1/H_X**2, 1/H_Y**2, 1/0.02**2]),  # step 3: h_z = 0.02
    np.diag([1/H_X**2, 1/H_Y**2, 1/H_Z**2])   # step 4: h_z = 0.01 (target)
    ]

    current = initial_path

    for step, M_stage in enumerate(STAGES, start=1):
        h_z_stage = 1.0 / M_stage[2, 2]**0.5
        print(f"\n--- Stage {step}/4:  h_z = {h_z_stage:.4f} ---")
        out = p(f"cube_adapted_step{step}.mesh")
        adapt_mesh(current, out, M_stage)
        to_vtu(out)
        current = out

if __name__ == "__main__":
    main()