# pyright: reportMissingImports=false, reportMissingModuleSource=false
import argparse
import csv
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from adaptive_algo import run_adaptive_poisson, PPRConvergenceTracker
from problems import get_problem, get_grad_exact
from solver import solve_poisson_generic

# Boundary-layer parameters for PPRConvergenceTracker, keyed by problem name.
# layer_dir: coordinate index whose variation defines the layer;
# layer_half_width: half-width of the region used to sample h_z.
_PPR_LAYER_PARAMS = {
    "1d":         dict(layer_dir=0, layer_centre=0.0, layer_half_width=0.3),
    "plan":       dict(layer_dir=0, layer_centre=0.0, layer_half_width=0.05),
    "sphere":     dict(layer_dir=0, layer_centre=0.5, layer_half_width=0.1),
    "tok-sphere": dict(layer_dir=2, layer_centre=400.0, layer_half_width=40.0),
    "tok-wall":   dict(layer_dir=0, layer_centre=500.0, layer_half_width=150.0),
}


def parse_args():
    p = argparse.ArgumentParser(description="Anisotropic adaptive Poisson solver")
    p.add_argument("--problem",  default="1d",   choices=["1d", "sphere", "plan", "tok-sphere", "tok-wall"],
                   help="Problem type (default: 1d)")
    p.add_argument("--k",        type=int, default=1, choices=[1, 2],
                   help="FE degree and estimator: 1=ZZ, 2=Naga-Zhang (default: 1)")
    p.add_argument("--results",  default=None,
                   help="Results directory name under poisson_cube/ (default: results_<problem>_k<k>)")
    p.add_argument("--mesh",     default=None,
                   help="Starting .mesh file; omit to generate from scratch")
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
    p.add_argument("--nosurf",   action="store_true", default=False,
                   help="Pass -nosurf to MMG3D (preserve surface mesh, needed for tokamak)")
    p.add_argument("--mmg-extra", default="", dest="mmg_extra",
                   help="Additional MMG3D flags as a quoted string, e.g. \"-hausd 5.0 -ar 21\"")
    # Tokamak wall snapping
    p.add_argument("--snap-walls", action="store_true", default=False,
                   help="After each MMG3D step, snap boundary vertices onto the exact cylindrical walls")
    p.add_argument("--snap-r-inner",    type=float, default=200.0, dest="snap_r_inner",
                   help="Exact inner cylindrical wall radius for snapping (default: 200.0)")
    p.add_argument("--snap-r-outer",    type=float, default=800.0, dest="snap_r_outer",
                   help="Exact outer cylindrical wall radius for snapping (default: 800.0)")
    return p.parse_args()


def main():
    args = parse_args()

    results_name = args.results or f"results_{args.problem}_k{args.k}"
    results_dir  = Path(__file__).resolve().parent / results_name
    results_dir.mkdir(exist_ok=True, parents=True)

    tol_values = [args.tol_start / (2**i) for i in range(args.n_tol)]

    # Write a human-readable summary of all run parameters.
    run_info_path = results_dir / "run_info.txt"
    mmg_extra_str = args.mmg_extra if args.mmg_extra else "(none)"
    if args.nosurf:
        mmg_extra_str = ("-nosurf " + mmg_extra_str).strip()
    tol_list_str = "  ".join(str(t) for t in tol_values)
    with open(run_info_path, "w") as fh:
        fh.write("Run configuration\n")
        fh.write("=" * 40 + "\n")
        fh.write(f"problem:           {args.problem}\n")
        fh.write(f"k (estimator):     {args.k}  ({'ZZ' if args.k == 1 else 'Naga-Zhang'})\n")
        fh.write(f"starting mesh:     {args.mesh if args.mesh else '(generated from scratch)'}\n")
        fh.write(f"tol_start:         {args.tol_start}\n")
        fh.write(f"n_tol:             {args.n_tol}\n")
        fh.write(f"tolerances:        {tol_list_str}\n")
        fh.write(f"n_loop:            {args.n_loop}\n")
        fh.write(f"hmin:              {args.hmin}\n")
        fh.write(f"hmax:              {args.hmax}\n")
        fh.write(f"hgrad:             {args.hgrad}\n")
        fh.write(f"alpha:             {args.alpha}\n")
        fh.write(f"correction_factor: {args.correction_factor}\n")
        fh.write(f"mmg3d:             {args.mmg3d}\n")
        fh.write(f"mmg_extra_args:    {mmg_extra_str}\n")
        fh.write(f"snap_walls:        {args.snap_walls}\n")
        if args.snap_walls:
            fh.write(f"snap_r_inner:      {args.snap_r_inner}\n")
            fh.write(f"snap_r_outer:      {args.snap_r_outer}\n")
    print(f"Run info saved → {run_info_path}")

    f_factory, g_np, u_exact = get_problem(args.problem)

    def make_solver(msh):
        return solve_poisson_generic(msh, f_factory(msh), g_np, args.k)

    mmg_extra: list | None = []
    if args.nosurf:
        mmg_extra.append("-nosurf")
    if args.mmg_extra:
        mmg_extra.extend(args.mmg_extra.split())
    mmg_extra = mmg_extra or None

    tok_snap = None
    if args.snap_walls:
        tok_snap = {
            "r_inner": args.snap_r_inner,
            "r_outer": args.snap_r_outer,
        }

    prev_mesh_file    = args.mesh   # None → generate from scratch on first TOL
    all_iter_metrics  = {}
    all_final_metrics = {}

    # PPR convergence tracker — only active for k=2 (Naga-Zhang estimator)
    ppr_tracker = None
    if args.k == 2:
        from nz_eta_estimatorP2 import Gh as _Gh
        _grad_ex = get_grad_exact(args.problem)
        _layer_kw = _PPR_LAYER_PARAMS.get(
            args.problem,
            dict(layer_dir=0, layer_centre=0.0, layer_half_width=0.3),
        )
        ppr_tracker = PPRConvergenceTracker(**_layer_kw)

    for tol in tol_values:
        print(f"\n{'#' * 70}")
        print(f"# problem={args.problem}  k={args.k}  TOL={tol}")
        print(f"{'#' * 70}")

        tol_dir = results_dir / f"tol_{tol}"

        msh, u_h, h_p, final_metrics, iter_metrics = run_adaptive_poisson(
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
            mmg_extra_args    = mmg_extra,
            tok_snap          = tok_snap,
        )

        prev_mesh_file = tol_dir / "meshes" / f"mesh_{args.n_loop - 1}.mesh"

        all_iter_metrics[tol]  = iter_metrics
        all_final_metrics[tol] = final_metrics

        if ppr_tracker is not None:
            ppr_tracker.record(tol, msh, u_h, _Gh, _grad_ex, h_p=h_p)

        print(f"\nFinal metrics for TOL={tol}:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value}")

    csv_path = results_dir / "convergence.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tol", "loop_idx", "n_vertices", "n_cells", "TRE", "max_aspect_ratio", "avg_aspect_ratio", "n_coarsen", "n_refine", "n_cells_coarsen", "n_cells_refine"])
        for tol in tol_values:
            for entry in all_iter_metrics[tol]:
                writer.writerow([
                    tol, entry["loop_idx"], entry["n_vertices"], entry["n_cells"], entry["TRE"],
                    entry["max_aspect_ratio"], entry["avg_aspect_ratio"],
                    entry["n_coarsen"], entry["n_refine"],
                    entry["n_cells_coarsen"], entry["n_cells_refine"],
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

    if ppr_tracker is not None:
        ppr_tracker.print_table()
        ppr_tracker.plot(str(results_dir / "ppr_convergence.pdf"))


if __name__ == "__main__":
    main()
