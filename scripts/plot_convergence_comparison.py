"""
Overlay the validation curves of two or more training runs.

    python scripts/plot_convergence_comparison.py \
        --run "Restormer=baseline_restormer" \
        --run "Restormer + SeeCoder=seecoder"

Draws val PSNR / SSIM / LPIPS against epoch, one line per run, with each run's
best epoch marked.

WHAT THIS FIGURE IS NOT
-----------------------
These are INDEPENDENT training runs, not a paired or controlled trajectory. They
share a seed and a configuration, but they differ in architecture, so from the
first optimizer step the RNG stream is consumed differently and the two runs
follow unrelated stochastic paths. cuDNN autotuning and non-deterministic kernels
add further divergence (`deterministic: false`).

Consequently:

* A gap between the curves at any single epoch is NOT a controlled measurement
  of the module's effect. Two reruns of the SAME configuration would also differ,
  and by how much is UNMEASURED in this repository -- no seed-repeat study exists.
* The curves show convergence BEHAVIOUR -- shape, stability, whether the schedule
  was exhausted -- which is what they are for.
* The quantitative claim lives in the paired per-image test-set comparison
  (`scripts/compare_paired_metrics.py`), not here.

The caption drawn onto the figure says this, so the figure cannot be lifted into
a slide deck without it.

Reads training CSVs READ-ONLY and touches no GPU, so it is safe to run against a
live experiment; an in-progress run simply plots as a shorter line.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.research import (  # noqa: E402
    apply_style, base_provenance, best_epoch, read_train_log, resolve_train_log,
    save_figure, series,
)

PANELS = [
    ("val_psnr", "Validation PSNR (dB)", "max"),
    ("val_ssim", "Validation SSIM", "max"),
    ("val_lpips", "Validation LPIPS", "min"),
]

CAPTION = (
    "Independent training runs, not a paired trajectory: identical seed and "
    "configuration, but different architectures consume the RNG stream "
    "differently.\nRun-to-run variance is UNMEASURED, so a gap between curves at "
    "one epoch is not a controlled measurement of the module's effect.\n"
    "Quantitative comparison: paired per-image test-set statistics, not this figure."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run", action="append", required=True,
                    help="LABEL=experiment (repeatable); experiment may be a name, "
                         "a results dir, or a train_log.csv path")
    ap.add_argument("--out-dir", default="research/figures")
    ap.add_argument("--stem", default="convergence_comparison")
    ap.add_argument("--formats", default="png,pdf")
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        if "=" not in spec:
            print(f"--run needs LABEL=experiment, got {spec!r}")
            return 2
        label, experiment = spec.split("=", 1)
        try:
            path = resolve_train_log(experiment)
        except FileNotFoundError as exc:
            print(exc)
            return 2
        rows = read_train_log(path)
        if not rows:
            print(f"{label}: {path} has no complete epochs yet")
            return 2
        runs.append({"label": label, "experiment": experiment, "log": path, "rows": rows})
        print(f"{label:<28} {path}  ({len(rows)} epochs)")

    apply_style()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(5.0 * len(PANELS), 4.4))
    if len(PANELS) == 1:
        axes = [axes]

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    provenance_runs = []

    for r_i, run in enumerate(runs):
        color = colors[r_i % len(colors)]
        rec = {"label": run["label"], "experiment": run["experiment"],
               "log": run["log"], "epochs_logged": len(run["rows"]), "best": {}}

        for ax, (col, ylabel, mode) in zip(axes, PANELS):
            xs, ys = series(run["rows"], col)
            if not xs:
                continue
            ax.plot(xs, ys, color=color, linewidth=1.6, label=run["label"])

            be, bv = best_epoch(run["rows"], col, mode)
            if be is not None:
                ax.plot([be], [bv], marker="o", markersize=6, color=color,
                        markeredgecolor="white", markeredgewidth=1.0, zorder=5)
                rec["best"][col] = {"epoch": be, "value": bv}
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

        provenance_runs.append(rec)

    for ax, (col, ylabel, mode) in zip(axes, PANELS):
        arrow = "higher is better" if mode == "max" else "lower is better"
        ax.set_title(f"{ylabel}  ({arrow})", fontsize=10)
    axes[0].legend(loc="best", fontsize=9)

    fig.suptitle("Convergence comparison — independent runs", fontsize=12)
    # Reserve the bottom strip for the caption; without this the text overlaps
    # the x-axis labels at small figure sizes.
    fig.tight_layout(rect=(0, 0.16, 1, 0.95))
    fig.text(0.5, 0.015, CAPTION, ha="center", va="bottom", fontsize=7.6,
             color="#444444", linespacing=1.45)

    prov = base_provenance(
        figure="convergence_comparison",
        runs=provenance_runs,
        panels=[p[0] for p in PANELS],
        note=("Independent runs. Marked points are each run's own best epoch. "
              "Run-to-run variance is unmeasured; curve separation is not a "
              "controlled effect measurement."),
    )
    written = save_figure(fig, args.out_dir, args.stem, prov,
                          tuple(f.strip() for f in args.formats.split(",") if f.strip()))
    plt.close(fig)

    for w in written:
        print(f"wrote {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
