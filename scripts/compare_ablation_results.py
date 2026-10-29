"""
Ablation comparison plots (Phase 9).

Reads every `results/<ablation>/summary.json` that evaluate.py has written and
draws PSNR / SSIM / LPIPS across the ablation ladder:

    1  Restormer                                  baseline_restormer
    2  + SeeCoder                                 seecoder
    3  + DiffAE                                   seecoder_diffae
    4  + LCR                                      seecoder_diffae_lcr
    5  + Rectified Flow                           seecoder_diffae_lcr_flow
    6  + SRA (full SRRT)                          full_srrt

    python scripts/compare_ablation_results.py
    python scripts/compare_ablation_results.py --results-dir results --out-dir research/figures

This script RUNS NO TRAINING and RUNS NO EVALUATION. It only reads what has
already been measured.

TWO INTEGRITY RULES ENFORCED HERE
---------------------------------
1. An ablation with no summary.json is drawn as a hatched "not run" slot, never
   omitted and never interpolated. Silently dropping unrun rungs makes a partial
   ladder look complete.
2. All present results must share one evaluation protocol (split, crop mode,
   crop size, image count) or the script exits. Mixing 256px centre crops with
   full-image evaluation in one chart would manufacture a multi-dB "improvement"
   out of a protocol difference.

Error bars, when drawn, are the per-image standard deviation from summary.json.
That is the spread ACROSS TEST IMAGES, not a confidence interval on the mean and
not a run-to-run seed variance -- the axis label says so, because a reader will
otherwise assume non-overlap implies significance.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.research import (  # noqa: E402
    ABLATION_ORDER, MODEL_COLORS, MODEL_LABELS, apply_style, assert_comparable,
    base_provenance, load_ablation_results, save_figure,
)

METRICS = [("psnr", "PSNR (dB)", "higher"),
           ("ssim", "SSIM", "higher"),
           ("lpips", "LPIPS", "lower")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="research/figures")
    ap.add_argument("--stem", default="ablation_comparison")
    ap.add_argument("--errorbars", action="store_true",
                    help="draw per-image std as error bars (spread, not significance)")
    ap.add_argument("--formats", default="png,pdf")
    args = ap.parse_args()

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    results = load_ablation_results(args.results_dir)
    protocol = assert_comparable(results)

    present = [n for n in ABLATION_ORDER if results.get(n)]
    missing = [n for n in ABLATION_ORDER if not results.get(n)]
    print(f"measured: {present or 'none'}")
    print(f"not run : {missing or 'none'}")
    if not present:
        print("\nNothing to plot yet: no ablation has been evaluated.\n"
              "Run evaluate.py on a checkpoint first, e.g.\n"
              "  python evaluate.py --checkpoint checkpoints/baseline_restormer/best_psnr.pth")
        return 1

    apply_style()
    import matplotlib.pyplot as plt

    labels = [MODEL_LABELS.get(n, n) for n in ABLATION_ORDER]
    xs = np.arange(len(ABLATION_ORDER))

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    table_values = {}

    for ax, (key, label, direction) in zip(axes, METRICS):
        vals, errs, colors, hatches = [], [], [], []
        for name in ABLATION_ORDER:
            s = results.get(name)
            m = (s or {}).get("metrics", {}).get(key)
            if m is None:
                vals.append(0.0)
                errs.append(0.0)
                colors.append("#FFFFFF")
                hatches.append("///")
            else:
                vals.append(float(m["mean"]))
                errs.append(float(m.get("std", 0.0)))
                colors.append(MODEL_COLORS.get(name, "#0072B2"))
                hatches.append("")
        table_values[key] = vals

        bars = ax.bar(xs, vals, color=colors, edgecolor="0.35", linewidth=0.9,
                      yerr=(errs if args.errorbars else None),
                      capsize=3 if args.errorbars else 0,
                      error_kw={"ecolor": "0.4", "elinewidth": 0.9})
        for b, h in zip(bars, hatches):
            if h:
                b.set_hatch(h)

        # Bars start at the smallest measured value minus headroom rather than at
        # zero: PSNR differences of ~1 dB are invisible on a 0-30 axis. The axis
        # is labelled with real numbers and is not inverted or clipped, so this
        # zooms without distorting ordering.
        measured = [v for v, hh in zip(vals, hatches) if not hh]
        if measured and key == "psnr":
            lo, hi = min(measured), max(measured)
            pad = max(0.5, (hi - lo) * 0.35)
            ax.set_ylim(max(0.0, lo - pad), hi + pad)

        for x, v, hh in zip(xs, vals, hatches):
            ax.text(x, v, "not run" if hh else f"{v:.4g}", ha="center",
                    va="bottom", fontsize=7.5, rotation=90 if hh else 0,
                    color="0.35" if hh else "black")

        arrow = r" ($\uparrow$ better)" if direction == "higher" else r" ($\downarrow$ better)"
        ax.set_title(label + arrow)
        ax.set_ylabel(label)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30, ha="right")

    proto_txt = (f"split={protocol[0]}, crop_mode={protocol[1]}, "
                 f"crop_size={protocol[2]}, n={protocol[3]} images"
                 if protocol else "no results")
    sub = f"Evaluation protocol (identical for every bar): {proto_txt}"
    if args.errorbars:
        sub += "\nError bars = per-image std across the test set (spread, NOT a " \
               "confidence interval and NOT run-to-run variance)."
    if missing:
        sub += f"\nHatched = not run: {', '.join(MODEL_LABELS.get(m, m) for m in missing)}"
    fig.suptitle("SRRT ablation ladder — measured results only\n" + sub, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    prov = base_provenance(
        figure="ablation_comparison",
        results_dir=os.path.abspath(args.results_dir),
        protocol={"split": protocol[0] if protocol else None,
                  "crop_mode": protocol[1] if protocol else None,
                  "crop_size": protocol[2] if protocol else None,
                  "num_images": protocol[3] if protocol else None},
        measured=present, not_run=missing,
        sources={n: results[n]["_summary_path"] for n in present},
        checkpoints={n: results[n].get("checkpoint") for n in present},
        epochs={n: results[n].get("epoch") for n in present},
        note="Unrun ablations are drawn hatched at zero and labelled 'not run'; "
             "no value is interpolated.",
    )
    written = save_figure(fig, args.out_dir, args.stem, prov, formats)
    plt.close(fig)

    for w in written:
        print(f"wrote {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
