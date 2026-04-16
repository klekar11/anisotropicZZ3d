# pyright: reportMissingImports=false, reportMissingModuleSource=false
import argparse
import csv
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from adaptive_algo import run_adaptive_poisson
from problems import get_problem
from solver import solve_poisson_generic


def parse_args():
    p = argparse.ArgumentParser(description="Anisotropic adaptive Poisson solver")
    p.add_argument("--problem",  default="1d",   choices=["1d", "sphere"],
                   help="Problem type (default: 1d)")
    p.add_argument("--k",        type=int, default=1, choices=[1, 2],
                   help="FE degree and estimator: 1=ZZ, 2=Naga-Zhang (default: 1)")
    p.add_argument("--results",  default=None,
                   help="Results directory name under poisson_cube/ (default: results_<problem>_k<k>)")
    p.add_argument("--n-loop",   type=int,   default=40)
    p.add_argument("--tol-start",type=float, default=1.0,
                   help="Starting tolerance (default: 1.0)")
    p.add_argument("--n-tol",    type=int,   default=6,
                   help="Number of tolerance halvings (default: 6)")
    p.add_argument("--hmax",     type=float, default=1.0)
    p.add_argument("--hmin",     type=float, default=1e-10)
    p.add_argument("--hgrad",    type=float, default=-1)
    p.add_argument("--alpha",    type=float, default=0.25)
    p.add_argument("--correction-factor", type=float, default=1.5)
    p.add_argument("--mmg3d",    default="/usr/local/bin/mmg3d_O3",
                   help="Path to mmg3d executable")
    return p.parse_args()


def main():
    args = parse_args()

    results_name = args.results or f"results_{args.problem}_k{args.k}"
    results_dir  = Path(__file__).resolve().parent / results_name
    results_dir.mkdir(exist_ok=True, parents=True)

    tol_values = [args.tol_start / (2**i) for i in range(args.n_tol)]

    f_factory, g_np, u_exact = get_problem(args.problem)

    def make_solver(msh):
        return solve_poisson_generic(msh, f_factory(msh), g_np, args.k)

    prev_mesh_file    = None
    all_iter_metrics  = {}
    all_final_metrics = {}

    for tol in tol_values:
        print(f"\n{'#' * 70}")
        print(f"# problem={args.problem}  k={args.k}  TOL={tol}")
        print(f"{'#' * 70}")

        tol_dir = results_dir / f"tol_{tol}"

        _, _, _, final_metrics, iter_metrics = run_adaptive_poisson(
            results_dir       = tol_dir,
            solver            = make_solver,
            u_numpy           = u_exact,
            n_loop            = args.n_loop,
            hmax              = args.hmax,
            hmin              = args.hmin,
            hgrad             = args.hgrad,
            tol               = tol,
            alpha             = args.alpha,
            correction_factor = args.correction_factor,
            mmg3d_exe         = args.mmg3d,
            initial_mesh_file = prev_mesh_file,
            k                 = args.k,
        )

        prev_mesh_file = tol_dir / "meshes" / f"mesh_{args.n_loop - 1}.mesh"

        all_iter_metrics[tol]  = iter_metrics
        all_final_metrics[tol] = final_metrics

        print(f"\nFinal metrics for TOL={tol}:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value}")

    csv_path = results_dir / "convergence.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tol", "loop_idx", "n_vertices", "TRE", "max_aspect_ratio", "avg_aspect_ratio"])
        for tol in tol_values:
            for entry in all_iter_metrics[tol]:
                writer.writerow([
                    tol, entry["loop_idx"], entry["n_vertices"], entry["TRE"],
                    entry["max_aspect_ratio"], entry["avg_aspect_ratio"],
                ])
    print(f"\nConvergence CSV saved → {csv_path}")

    txt_path = results_dir / "final_metrics_summary.txt"
    with open(txt_path, "w") as fh:
        fh.write(f"problem: {args.problem}\n")
        fh.write(f"k: {args.k}\n\n")
        for tol in tol_values:
            m = all_final_metrics[tol]
            fh.write(f"TOL: {tol}\n")
            for key, value in m.items():
                fh.write(f"  {key}: {value}\n")
            fh.write("\n")
    print(f"Final metrics summary saved → {txt_path}")


if __name__ == "__main__":
    main()
