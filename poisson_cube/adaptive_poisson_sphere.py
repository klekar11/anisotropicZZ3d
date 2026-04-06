# pyright: reportMissingImports=false, reportMissingModuleSource=false
import csv
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Add parent directory to sys.path so sibling modules are importable
# ---------------------------------------------------------------------------
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
	sys.path.insert(0, str(root))

from adaptive_algo import run_adaptive_poisson  # noqa: E402
from solver import solve_sphere_poisson, u_exact_sphere_np  # noqa: E402

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_LOOP = 10
HMAX = 1.0
HMIN = 1e-10
HGRAD = -1
ALPHA = 0.25
CORRECTION_FACTOR = 1.5
MMG3D_EXE = "/usr/local/bin/mmg3d_O3"

# Sphere profile parameters
R = 0.5
EPSILON = 0.02

# Tolerance sequence: start at 1 and halve four times
TOL_VALUES = [1.0, 0.5, 0.25, 0.125]

results_dir = Path(__file__).resolve().parent / "results_sphere"
results_dir.mkdir(exist_ok=True, parents=True)


def sphere_u_numpy(x: np.ndarray) -> np.ndarray:
	return u_exact_sphere_np(x, R=R, epsilon=EPSILON)


def sphere_solver(msh):
	return solve_sphere_poisson(msh, degree=1, R=R, epsilon=EPSILON)


# ---------------------------------------------------------------------------
# Multi-tolerance adaptive loop
#
# For TOL=1.0 the initial mesh is generated from scratch.
# For every subsequent TOL the last mesh produced by the previous run is
# copied into the new TOL directory as mesh_0.mesh, so adaptation continues
# from where the previous tolerance left off.
# ---------------------------------------------------------------------------
prev_mesh_file = None   # path to last .mesh from the previous TOL run
all_iter_metrics = {}   # tol -> list[{loop_idx, n_vertices, TRE}]
all_final_metrics = {}  # tol -> final_metrics dict

for tol in TOL_VALUES:
	print(f"\n{'#' * 70}")
	print(f"# TOL = {tol}")
	print(f"{'#' * 70}")

	tol_dir = results_dir / f"tol_{tol}"

	_, _, _, final_metrics, iter_metrics = run_adaptive_poisson(
		results_dir=tol_dir,
		solver=sphere_solver,
		u_numpy=sphere_u_numpy,
		n_loop=N_LOOP,
		hmax=HMAX,
		hmin=HMIN,
		hgrad=HGRAD,
		tol=tol,
		alpha=ALPHA,
		correction_factor=CORRECTION_FACTOR,
		mmg3d_exe=MMG3D_EXE,
		initial_mesh_file=prev_mesh_file,
	)

	# The last mesh solved on is mesh_{N_LOOP-1}.mesh inside this TOL directory
	prev_mesh_file = tol_dir / "meshes" / f"mesh_{N_LOOP - 1}.mesh"

	all_iter_metrics[tol] = iter_metrics
	all_final_metrics[tol] = final_metrics

	print(f"\nFinal error metrics for TOL={tol}:")
	for key, value in final_metrics.items():
		print(f"  {key}: {value}")

# ---------------------------------------------------------------------------
# CSV: one row per (TOL, iteration) with n_vertices and TRE
# Stored directly in results_dir for easy plotting.
# ---------------------------------------------------------------------------
csv_path = results_dir / "convergence.csv"
with open(csv_path, "w", newline="") as fh:
	writer = csv.writer(fh)
	writer.writerow(["tol", "loop_idx", "n_vertices", "TRE"])
	for tol in TOL_VALUES:
		for entry in all_iter_metrics[tol]:
			writer.writerow([tol, entry["loop_idx"], entry["n_vertices"], entry["TRE"]])
print(f"\nConvergence CSV saved -> {csv_path}")

# ---------------------------------------------------------------------------
# Text summary: final error metrics for each TOL
# Stored directly in results_dir.
# ---------------------------------------------------------------------------
txt_path = results_dir / "final_metrics_summary.txt"
with open(txt_path, "w") as fh:
	fh.write(f"R: {R}\n")
	fh.write(f"EPSILON: {EPSILON}\n\n")
	for tol in TOL_VALUES:
		m = all_final_metrics[tol]
		fh.write(f"TOL: {tol}\n")
		for key, value in m.items():
			fh.write(f"  {key}: {value}\n")
		fh.write("\n")
print(f"Final metrics summary saved -> {txt_path}")
