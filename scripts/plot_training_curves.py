"""
Publication-quality training curves for any experiment (Phase 2).

Supersedes `scripts/plot_training.py`, which produced a single low-DPI PNG with
no provenance and no best-epoch marker. That script is kept working for the
existing 10-epoch artifacts; new figures should come from this one.

Reads the training CSV READ-ONLY, so it is safe to run against a live run:

    python scripts/plot_training_curves.py --experiment baseline_restormer
    python scripts/plot_training_curves.py --experiment results/restormer_baseline
    python scripts/plot_training_curves.py --experiment logs/x/train_log.csv --individual

Outputs (default --out-dir research/figures):

    <name>_training.png / .pdf / .json     6-panel combined figure
    <name>_training_<metric>.png / .pdf    individual panels (--individual)

Axis policy: y-limits are left to matplotlib and never clamped to exaggerate a
trend. `--no-zero-base` is deliberately NOT offered for the loss panels.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.research import (  # noqa: E402
    METRIC_COLORS, apply_style, base_provenance, best_epoch, read_train_log,
    resolve_train_log, save_figure, series,
)

# (csv column, axis label, better-direction, best-mode). Order = panel order.
PANELS = [
    ("train_loss", "Training loss", None, None),
    ("val_loss", "Validation loss", "lower", "min"),
    ("val_psnr", "Validation PSNR (dB)", "higher", "max"),
    ("val_ssim", "Validation SSIM", "higher", "max"),
    ("val_lpips", "Validation LPIPS", "lower", "min"),
    ("lr", "Learning rate", None, None),
]


def derive_stem(log_path: str) -> str:
    """
    Name the figure after the experiment, not after the file.

    `logs/baseline_restormer/train_log.csv`          -> baseline_restormer
    `results/restormer_baseline/metrics/train_log.csv` -> restormer_baseline
    """
    parent = os.path.dirname(os.path.abspath(log_path))
    if os.path.basename(parent) == "metrics":
        parent = os.path.dirname(parent)
    return os.path.basename(parent) or "experiment"


def draw_panel(ax, rows, column, label, direction, mode, mark_epoch):
    xs, ys = series(rows, column)
    if not xs:
        ax.text(0.5, 0.5, f"{column}\nnot recorded", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="0.45")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label)
        return None

    ax.plot(xs, ys, color=METRIC_COLORS.get(column, "#0072B2"), marker="o")

    # The LR spans 2e-4 -> 1e-6; on a linear axis the tail is an invisible flat
    # line, which would hide whether the cosine actually completed.
    if column == "lr":
        ax.set_yscale("log")

    best = None
    if mode:
        be, bv = best_epoch(rows, column, mode)
        if be is not None:
            best = (be, bv)
            ax.plot([be], [bv], marker="*", markersize=11, linestyle="none",
                    color="#D55E00", zorder=5,
                    label=f"best: {bv:.4g} @ ep {be}")
            ax.legend(loc="best")

    # Vertical marker at the best-val-PSNR epoch on EVERY panel, so the reader can
    # read off what the selected checkpoint scored on the other metrics too.
    if mark_epoch is not None:
        ax.axvline(mark_epoch, color="0.35", linestyle="--", linewidth=0.9, zorder=1)

    arrow = {"higher": r"  ($\uparrow$)", "lower": r"  ($\downarrow$)"}.get(direction, "")
    ax.set_title(label + arrow)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--experiment", default="baseline_restormer",
                    help="experiment name, results dir, or path to a train_log.csv")
    ap.add_argument("--out-dir", default="research/figures")
    ap.add_argument("--name", default=None,
                    help="output filename stem (default: derived from --experiment)")
    ap.add_argument("--individual", action="store_true",
                    help="also write one high-resolution figure per metric")
    ap.add_argument("--formats", default="png,pdf")
    args = ap.parse_args()

    log_path = resolve_train_log(args.experiment)
    rows = read_train_log(log_path)
    if len(rows) < 2:
        print(f"{log_path}: need >= 2 completed epochs to plot, found {len(rows)}.")
        return 1

    stem = args.name or derive_stem(log_path)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    apply_style()
    import matplotlib.pyplot as plt

    psnr_epoch, psnr_value = best_epoch(rows, "val_psnr", "max")
    epochs = [int(float(r["epoch"])) for r in rows]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    bests = {}
    for ax, (col, label, direction, mode) in zip(axes.ravel(), PANELS):
        bests[col] = draw_panel(ax, rows, col, label, direction, mode, psnr_epoch)

    title = f"{stem} — training diagnostics ({len(rows)} epochs logged)"
    if psnr_epoch is not None:
        title += f"\ndashed line = best validation PSNR (epoch {psnr_epoch}, {psnr_value:.3f} dB)"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    prov = base_provenance(
        figure="training_curves",
        experiment=stem,
        source_log=os.path.abspath(log_path),
        epochs_logged=len(rows),
        epoch_range=[min(epochs), max(epochs)],
        best={k: (None if v is None else {"epoch": v[0], "value": v[1]})
              for k, v in bests.items()},
        note="Log read read-only; a torn trailing line, if any, was dropped. "
             "No axis limits were clamped.",
    )

    written = save_figure(fig, args.out_dir, f"{stem}_training", prov, formats)
    plt.close(fig)

    if args.individual:
        for col, label, direction, mode in PANELS:
            f1, a1 = plt.subplots(figsize=(5.2, 3.6))
            draw_panel(a1, rows, col, label, direction, mode, psnr_epoch)
            f1.tight_layout()
            p = base_provenance(figure=f"training_{col}", experiment=stem,
                                source_log=os.path.abspath(log_path),
                                epochs_logged=len(rows))
            written += save_figure(f1, args.out_dir, f"{stem}_training_{col}", p, formats)
            plt.close(f1)

    for w in written:
        print(f"wrote {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
