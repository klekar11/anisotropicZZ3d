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
    "A": {"label": r"$A=\|\nabla u_h-\nabla u\|$",
          "color": "#1f77b4", "marker": "o", "ls": "-"},
    "B": {"label": r"$B=\|\nabla u_h-G_h u_h\|$",
          "color": "#ff7f0e", "marker": "s", "ls": "-"},
    "C": {"label": r"$C=\|G_h u_h-\nabla u\|$",
          "color": "#2ca02c", "marker": "^", "ls": "-"},
    "D": {"label": r"$D=\|\nabla u-G_h(I^2 u)\|$",
          "color": "#d62728", "marker": "D", "ls": "--"},
    "E": {"label": r"$E=\|G_h u_h-G_h(I^2 u)\|$",
          "color": "#9467bd", "marker": "v", "ls": "--"},
    "F": {"label": r"$F=\|\nabla(I^2 u)-\nabla u_h\|$",
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
    return sorted(recs, key=lambda r: r["h_layer"])


def _make_fig(recs, h, subset):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in subset:
        base = name.replace("_int", "")
        info = _FIELDS[base]
        vals = np.array([r.get(name, np.nan) for r in recs])
        if np.all(np.isfinite(vals)) and np.all(vals > 0):
            label = info["label"] + (" (int)" if name.endswith("_int") else "")
            ax.loglog(h, vals, marker=info["marker"], linestyle=info["ls"],
                       color=info["color"], linewidth=2, markersize=7, label=label)
    if len(recs) >= 2:
        h_ref = np.array([h.min(), h.max()])
        a_key = "A_int" if subset[0].endswith("_int") else "A"
        d_key = "D_int" if subset[0].endswith("_int") else "D"
        A_fine = float(recs[-1].get(a_key, recs[-1].get("A", np.nan)))
        if np.isfinite(A_fine):
            ax.loglog(h_ref, (A_fine / h[-1] ** 2) * h_ref ** 2,
                      "--", color="gray", lw=1, alpha=0.5, label=r"$O(h^2)$")
        D_fine = float(recs[-1].get(d_key, recs[-1].get("D", np.nan)))
        if np.isfinite(D_fine) and D_fine > 0:
            ax.loglog(h_ref, (D_fine / h[-1] ** 3) * h_ref ** 3,
                      ":", color="gray", lw=1, alpha=0.5, label=r"$O(h^3)$")
    ax.set_xlabel(r"$h_z$", fontsize=13)
    ncol = 1 if len(subset) <= 3 else 2
    ax.legend(fontsize=9, loc="upper left", ncol=ncol)
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

    h = np.array([r["h_layer"] for r in recs])

    out = Path(out_path).resolve() if out_path else csv_path.resolve().with_suffix(".pdf")
    parent = out.parent
    stem = out.stem

    subsets = [
        (["A", "B", "C"], "_ABC"),
        (["D", "E", "F"], "_DEF"),
        (list(_FIELDS.keys()), "_all"),
    ]
    has_int = any("A_int" in r for r in recs)
    if has_int:
        subsets += [
            (["A_int", "B_int", "C_int"], "_ABC_int"),
            (["D_int", "E_int", "F_int"], "_DEF_int"),
            ([k + "_int" for k in _FIELDS], "_all_int"),
        ]

    for subset, suffix in subsets:
        fig = _make_fig(recs, h, subset)
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
