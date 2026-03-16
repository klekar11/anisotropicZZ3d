import os
import numpy as np
import subprocess
from cube_anisoZ import _signed_tet_volume, write_cube_mesh, read_num_vertices, run_mmg3d, to_vtu

MMG3D_EXE = "mmg3d_O3"
OUTPUT_DIR = "cube_bands_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

H_COARSE = 0.10
H_FINE   = 0.02

BAND_BOUNDARIES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BAND_TYPE = ["coarse", "fine", "coarse", "fine", "coarse"]

HMIN_GLOBAL = 1e-4
HMAX_GLOBAL = 2.0
HAUSD = H_FINE / 2.0
H_FINE_STAGES = [0.06, H_FINE]

def band_mesh_size_sharp(z):

    for k, btype in enumerate(BAND_TYPE):
        z_lo = BAND_BOUNDARIES[k]
        z_hi = BAND_BOUNDARIES[k + 1]
        if z_lo <= z <= z_hi:
            return H_FINE if btype == "fine" else H_COARSE
    # outside [0,1] — clamp to coarse
    return H_COARSE

def isotropic_metric(h):
    lam = 1.0 / h**2
    return np.diag([lam, lam, lam])

def read_vertices(mesh_path):

    coords = []
    with open(mesh_path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "Vertices":
            n = int(lines[i+1].strip())
            for j in range(n):
                parts = lines[i+2+j].split()
                # Each line: x y z ref  (ref is discarded)
                coords.append([float(parts[0]),
                                float(parts[1]),
                                float(parts[2])])
            break
        i += 1
    return np.array(coords)

def write_sol_file_bands(path, mesh_path, h_fine_current=H_FINE):
    
    coords = read_vertices(mesh_path)   # (N, 3)
    nv = len(coords)
 
    # temporarily override H_FINE with the current stage value
    def h_at_vertex(z):
        h = band_mesh_size_sharp(z)
        return h_fine_current if h == H_FINE else h
 
    with open(path, "w") as sf:
        sf.write("MeshVersionFormatted 2\nDimension 3\n\n")
        sf.write("SolAtVertices\n")
        sf.write(f"{nv}\n")
        sf.write("1 3\n")   # 1 field, type 3 = symmetric tensor (6 values in 3D)
 
        counts = {"coarse": 0, "fine": 0}
        for k, (x, y, z) in enumerate(coords):
            h   = h_at_vertex(z)
            lam = 1.0 / h**2
            # isotropic: M = lam * I  ->  6 components: lam 0 lam 0 0 lam
            sf.write(f"{lam:.10e} 0.0000000000e+00 {lam:.10e} "
                     f"0.0000000000e+00 0.0000000000e+00 {lam:.10e}\n")
 
        sf.write("End\n")
    

def generate_initial_mesh(coarse_path, output_path, hmax=0.12):

    run_mmg3d([
        "-in",   coarse_path,
        "-out",  output_path,
        "-hmax", hmax,
        "-hmin", HMIN_GLOBAL,
    ])
    print(f"[Phase 1] -> {output_path}  ({read_num_vertices(output_path)} vertices)")
    return output_path

def adapt_mesh_bands(input_path, output_path, h_fine_current):

    sol_path = output_path.replace(".mesh", ".sol")
    write_sol_file_bands(sol_path, input_path, h_fine_current)
 
    run_mmg3d([
        "-in",    input_path,
        "-sol",   sol_path,
        "-out",   output_path,
        "-hmin",  HMIN_GLOBAL,   # safety floor  (never reached)
        "-hmax",  HMAX_GLOBAL,   # safety ceiling (never reached)
        "-hausd", HAUSD,
    ])

    nv = read_num_vertices(output_path)
    print(f"[Adapt] -> {output_path}  ({nv} vertices)")
    return output_path

def main():
    p = lambda name: os.path.join(OUTPUT_DIR, name)
 
    print("=" * 65)
    print("Band structure (z direction, cube [0,1]^3):")
    for k, btype in enumerate(BAND_TYPE):
        h = H_FINE if btype == "fine" else H_COARSE
        print(f"  z in [{BAND_BOUNDARIES[k]:.1f}, {BAND_BOUNDARIES[k+1]:.1f}]  "
              f"{btype:6s}  h = {h}")
    print("Transitions: sharp step (MMG default hgrad=1.3 handles smoothing)")
    print("=" * 65)
 
    # ── Phase 1: uniform coarse starting mesh ──────────────────────────────
    print("\nPHASE 1 — Uniform initial mesh")
    coarse  = p("cube_coarse.mesh")
    initial = p("cube_initial.mesh")
    write_cube_mesh(coarse)
    generate_initial_mesh(coarse, initial, hmax=0.12)
    to_vtu(initial)
 
    # ── Phase 2: gradual band adaptation ───────────────────────────────────
    print("\nPHASE 2 — Gradual band adaptation")
    print(f"  h_fine ramp: {H_FINE_STAGES}")
 
    current = initial
    for step, h_fine_now in enumerate(H_FINE_STAGES, start=1):
        print(f"\n--- Stage {step}/4  h_fine = {h_fine_now} ---")
        out = p(f"cube_bands_step{step}.mesh")
        adapt_mesh_bands(current, out, h_fine_now)
        to_vtu(out)
        current = out

if __name__ == "__main__":
    main()
