"""
Plot training curves from logs/<experiment>/train_log.csv.

SUPERSEDED by scripts/plot_training_curves.py, which adds PDF output, 300 DPI,
best-epoch markers, a log-scale LR panel and a provenance sidecar. This script is
kept because the existing 10-epoch artifacts were generated with it; use the new
one for anything going into the paper.

Only plots columns that actually contain data -- a run with the SRRT modules
disabled has no flow/LCR terms, and inventing empty panels for them would be
misleading.

    python scripts/plot_training.py --experiment baseline_restormer
    python scripts/plot_training.py --experiment baseline_restormer --out results/baseline_restormer/curves.png
"""

import argparse
import csv
import os
import sys


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="baseline_restormer")
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    log = args.log or os.path.join("logs", args.experiment, "train_log.csv")
    if not os.path.exists(log):
        print(f"No log at {log}")
        return 1

    with open(log, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"{log} has no data rows yet.")
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(r["epoch"]) for r in rows]

    panels = [
        ("train_loss", "Training loss", "lower better"),
        ("val_loss", "Validation loss", "lower better"),
        ("val_psnr", "Validation PSNR (dB)", "higher better"),
        ("val_ssim", "Validation SSIM", "higher better"),
        ("val_lpips", "Validation LPIPS", "LOWER better"),
        ("lr", "Learning rate", ""),
        ("peak_vram_mb", "Peak VRAM (MB)", ""),
        ("peak_ram_mb", "Peak system RAM (MB)", ""),
        ("iters_per_s", "Throughput (it/s)", ""),
    ]
    # Keep only panels with at least one real value.
    live = []
    for key, title, note in panels:
        vals = [to_float(r.get(key)) for r in rows]
        if any(v is not None for v in vals):
            live.append((key, title, note, vals))

    ncol = 3
    nrow = (len(live) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), squeeze=False)

    for ax, (key, title, note, vals) in zip(axes.flat, live):
        xs = [e for e, v in zip(epochs, vals) if v is not None]
        ys = [v for v in vals if v is not None]
        ax.plot(xs, ys, marker="o", ms=3)
        ax.set_title(title + (f"  ({note})" if note else ""))
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if key == "lr":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    for ax in axes.flat[len(live):]:
        ax.axis("off")

    n = len(rows)
    fig.suptitle(f"{args.experiment} - {n} epoch(s) logged"
                 + ("  [NOT converged]" if n < 50 else ""))
    fig.tight_layout()

    out = args.out or os.path.join("results", args.experiment, "training_curves.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)

    print(f"Plotted {len(live)} panels over {n} epochs -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
