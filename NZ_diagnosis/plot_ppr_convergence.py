# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Generate PPR convergence figures from a CSV produced by
``PPRConvergenceTracker.to_csv``.

Usage
-----
    python NZ_diagnosis/plot_ppr_convergence.py results/ppr_convergence.csv \
        [-o results/ppr_convergence.pdf]

Reads the six A–F (and optional A_int..F_int) columns and writes the same
``_ABC``, ``_DEF``, ``_all`` (and interior-only) PDF/PNG figures that
``PPRConvergenceTracker.plot`` used to produce inline.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

_FIELDS = {
    "A": {"label": r"$\|\nabla u_h-\nabla u\|$",
          "color": "#1f77b4", "marker": "o", "ls": "-"},
    "B": {"label": r"$\|\nabla u_h-G_h u_h\|$",
          "color": "#ff7f0e", "marker": "s", "ls": "-"},
    "C": {"label": r"$\|G_h u_h-\nabla u\|$",
          "color": "#2ca02c", "marker": "^", "ls": "-"},
    "D": {"label": r"$\|\nabla u-G_h(I^2 u)\|$",
          "color": "#d62728", "marker": "D", "ls": "--"},
    "E": {"label": r"$\|G_h u_h-G_h(I^2 u)\|$",
          "color": "#9467bd", "marker": "v", "ls": "--"},
    "F": {"label": r"$\|\nabla(I^2 u)-\nabla u_h\|$",
          "color": "#8c564b", "marker": "P", "ls": "--"},
}


def _load_records(csv_path: "str | Path") -> list[dict]:
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        recs = []
        for row in reader:
            rec = {}
            for k, v in row.items():
                if v == "" or v is None:
                    continue
                rec[k] = float(v)
            recs.append(rec)
    return sorted(recs, key=lambda r: r["lambda3_min"])


def _make_fig(recs, n, subset):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in subset:
        base = name.replace("_int", "")
        info = _FIELDS[base]
        vals = np.array([r.get(name, np.nan) for r in recs])
        if np.all(np.isfinite(vals)) and np.all(vals > 0):
            label = info["label"] + (" (int)" if name.endswith("_int") else "")
            ax.loglog(n, vals, marker=info["marker"], linestyle=info["ls"],
                       color=info["color"], linewidth=2, markersize=7, label=label)
    if len(recs) >= 3:
        d_key = "D_int" if subset[0].endswith("_int") else "D"
        e_key = "E_int" if subset[0].endswith("_int") else "E"
        # Compute convergence rates from third-to-last to first point
        D_vals = np.array([r.get(d_key, np.nan) for r in recs])
        E_vals = np.array([r.get(e_key, np.nan) for r in recs])
        
        # D convergence line - mean rate from all consecutive pairs up to third-to-last
        if D_vals[0] > 0:
            rates_D = []
            for i in range(len(D_vals)-1):
                if np.isfinite(D_vals[i]) and np.isfinite(D_vals[i+1]) and D_vals[i] > 0 and D_vals[i+1] > 0 and n[i] > 0 and n[i+1] > 0:
                    rate = np.log(D_vals[i+1] / D_vals[i]) / np.log(n[i+1] / n[i])
                    rates_D.append(rate)
            if rates_D:
                mean_rate_D = np.mean(rates_D)
                n_line = n
                D_line = D_vals[0] * (n_line / n[0]) ** mean_rate_D
                ax.loglog(n_line, D_line, "--", color="black", lw=1.5, alpha=0.7, label=f"Rate: {mean_rate_D:.2f}")
        
        # E convergence line - mean rate from all consecutive pairs up to third-to-last
        if E_vals[0] > 0:
            rates_E = []
            for i in range(len(E_vals) - 3):
                if np.isfinite(E_vals[i]) and np.isfinite(E_vals[i+1]) and E_vals[i] > 0 and E_vals[i+1] > 0 and n[i] > 0 and n[i+1] > 0:
                    rate = np.log(E_vals[i+1] / E_vals[i]) / np.log(n[i+1] / n[i])
                    rates_E.append(rate)
            if rates_E:
                mean_rate_E = np.mean(rates_E)
                n_line = n
                E_line = E_vals[0] * (n_line / n[0]) ** mean_rate_E
                ax.loglog(n_line, E_line, ":", color="black", lw=1.5, alpha=0.7, label=f"Rate: {mean_rate_E:.2f}")
    ax.set_xlabel(r"$\lambda_{3,\mathrm{min}}$", fontsize=15)
    ncol = 1 if len(subset) <= 3 else 2
    ax.legend(fontsize=14, loc="best", ncol=ncol,
              handlelength=1.5, labelspacing=0.4, borderpad=0.6)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_ppr_convergence(csv_path: "str | Path", out_path: "str | Path | None" = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    recs = _load_records(csv_path)
    if len(recs) == 0:
        print(f"  [plot_ppr_convergence] No records in {csv_path} — nothing to plot.")
        return

    n = np.array([r["lambda3_min"] for r in recs])

    out = Path(out_path).resolve() if out_path else csv_path.resolve().with_suffix(".pdf")
    parent = out.parent
    stem = out.stem

    subsets = [
        (["A", "B", "C"], "_ABC"),
        (["D", "E"], "_DEF"),
        (list(_FIELDS.keys()), "_all"),
    ]
    has_int = any("A_int" in r for r in recs)
    if has_int:
        subsets += [
            (["A_int", "B_int", "C_int"], "_ABC_int"),
            (["D_int", "E_int"], "_DEF_int"),
            ([k + "_int" for k in _FIELDS], "_all_int"),
        ]

    for subset, suffix in subsets:
        fig = _make_fig(recs, n, subset)
        for ext in (".pdf", ".png"):
            p = parent / (stem + suffix + ext)
            fig.savefig(str(p), dpi=400, bbox_inches="tight")
            print(f"  [plot_ppr_convergence] Saved → {p}")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="CSV produced by PPRConvergenceTracker.to_csv")
    parser.add_argument("-o", "--output", default=None,
                         help="Base path for figures (default: csv_path with .pdf suffix)")
    args = parser.parse_args()
    plot_ppr_convergence(args.csv_path, args.output)


if __name__ == "__main__":
    main()
