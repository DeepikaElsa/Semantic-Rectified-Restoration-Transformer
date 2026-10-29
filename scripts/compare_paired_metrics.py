"""
Paired per-image comparison between two evaluated models.

Both models are scored on the SAME images by `evaluate.py`, so the comparison
can be PAIRED: the difference is computed per image and then aggregated, which
removes image difficulty as a source of variance and is far more sensitive than
comparing two independent means.

    python scripts/compare_paired_metrics.py \
        --baseline results/restormer_baseline \
        --ablation results/ablation_seecoder

Reports, per metric: mean / median / std of the per-image difference, a 95%
confidence interval on the mean, win-loss-tie counts, a paired t-test and a
Wilcoxon signed-rank test.

WHAT THE CONFIDENCE INTERVAL IS, AND IS NOT
-------------------------------------------
The CI is over TEST IMAGES, for one fixed pair of trained checkpoints. It answers:
"if I drew another sample of GoPro-like images, how much would the measured mean
difference between THESE TWO checkpoints move?"

It is **not** a training-run variance estimate. It says nothing about how much
the result would move if either model were retrained with a different seed, and
it therefore cannot on its own establish that the *method* helps. Each arm here
is a single training run, so run-to-run variance is unmeasured and a
statistically significant per-image difference may still be seed noise. Both the
JSON and the markdown carry this caveat so it cannot be read off without it.

This script RUNS NO TRAINING, NO EVALUATION and NO GPU WORK. It only reads CSVs
that have already been written, so it is safe to run while a job is training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# metric -> +1 if higher is better, -1 if lower is better
METRIC_DIRECTION = {"psnr": +1, "ssim": +1, "lpips": -1}

# Fields that must match between the two runs or the comparison is invalid.
PROTOCOL_FIELDS = ("split", "crop_mode", "crop_size", "num_images")


def _find(run_dir: str, name: str) -> str | None:
    """evaluate.py writes into run_dir; collect_baseline_results.py into run_dir/metrics."""
    for candidate in (os.path.join(run_dir, name),
                      os.path.join(run_dir, "metrics", name)):
        if os.path.isfile(candidate):
            return candidate
    return None


def read_metrics(run_dir: str) -> tuple[dict[str, dict[str, float]], dict]:
    """Return {filename: {metric: value}} and the run's summary.json."""
    mpath = _find(run_dir, "metrics.csv")
    spath = _find(run_dir, "summary.json")
    if mpath is None:
        raise SystemExit(f"no metrics.csv under {run_dir} -- run evaluate.py first")
    if spath is None:
        raise SystemExit(f"no summary.json under {run_dir} -- run evaluate.py first")

    rows: dict[str, dict[str, float]] = {}
    with open(mpath, newline="") as f:
        for r in csv.DictReader(f):
            name = r["filename"]
            vals = {}
            for m in METRIC_DIRECTION:
                if r.get(m) not in (None, "", "None"):
                    vals[m] = float(r[m])
            rows[name] = vals

    with open(spath) as f:
        summary = json.load(f)
    return rows, summary


def protocol_of(summary: dict) -> dict:
    return {k: summary.get(k) for k in PROTOCOL_FIELDS}


def t_critical(df: int, conf: float = 0.95) -> float:
    """Two-sided t critical value; falls back to the normal value without scipy."""
    try:
        from scipy import stats
        return float(stats.t.ppf(0.5 + conf / 2.0, df))
    except Exception:
        return 1.959963984540054  # z_{0.975}; df here is ~1000 so the gap is <0.2%


def bootstrap_ci(d: np.ndarray, iters: int = 10000, conf: float = 0.95,
                 seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean -- makes no normality assumption."""
    rng = np.random.default_rng(seed)
    n = len(d)
    means = d[rng.integers(0, n, size=(iters, n))].mean(axis=1)
    lo = float(np.percentile(means, 100 * (0.5 - conf / 2)))
    hi = float(np.percentile(means, 100 * (0.5 + conf / 2)))
    return lo, hi


def analyse(delta: np.ndarray, direction: int, seed: int = 42) -> dict:
    """Aggregate statistics for one metric's per-image differences."""
    n = int(len(delta))
    mean = float(delta.mean())
    # ddof=1: these images are a sample, not the population.
    std = float(delta.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    tcrit = t_critical(n - 1) if n > 1 else float("nan")
    ci = (mean - tcrit * sem, mean + tcrit * sem)

    improved = int((delta * direction > 0).sum())
    worsened = int((delta * direction < 0).sum())
    tied = int((delta == 0).sum())

    out = {
        "n_images": n,
        "mean": mean,
        "median": float(np.median(delta)),
        "std": std,
        "sem": sem,
        "ci95_mean": [ci[0], ci[1]],
        "ci95_method": "Student-t on per-image differences",
        "ci95_bootstrap": list(bootstrap_ci(delta, seed=seed)) if n > 1 else [float("nan")] * 2,
        "min": float(delta.min()),
        "max": float(delta.max()),
        "improved_images": improved,
        "worsened_images": worsened,
        "tied_images": tied,
        "direction": "higher_better" if direction > 0 else "lower_better",
        "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
    }

    try:
        from scipy import stats
        out["paired_t_p"] = float(stats.ttest_rel(delta, np.zeros_like(delta)).pvalue)
        if np.any(delta != 0):
            out["wilcoxon_p"] = float(stats.wilcoxon(delta).pvalue)
        # Cohen's d for paired samples: mean difference in units of its own SD.
        out["cohens_d_paired"] = float(mean / std) if std > 0 else float("nan")
    except Exception as exc:                                  # pragma: no cover
        out["tests_unavailable"] = repr(exc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--baseline", default="results/restormer_baseline",
                    help="directory holding the reference metrics.csv + summary.json")
    ap.add_argument("--ablation", default="results/ablation_seecoder",
                    help="directory holding the ablation metrics.csv + summary.json")
    ap.add_argument("--baseline-label", default="Restormer")
    ap.add_argument("--ablation-label", default="Restormer + SeeCoder")
    ap.add_argument("--out", default="research/tables/ablation_seecoder_paired.json")
    ap.add_argument("--markdown", default="research/tables/ablation_seecoder_paired.md")
    ap.add_argument("--seed", type=int, default=42, help="bootstrap seed")
    ap.add_argument("--allow-protocol-mismatch", action="store_true",
                    help="compare anyway (records the mismatch); off by default")
    args = ap.parse_args()

    base_rows, base_sum = read_metrics(args.baseline)
    abl_rows, abl_sum = read_metrics(args.ablation)

    pb, pa = protocol_of(base_sum), protocol_of(abl_sum)
    mismatch = {k: (pb[k], pa[k]) for k in PROTOCOL_FIELDS if pb[k] != pa[k]}
    if mismatch and not args.allow_protocol_mismatch:
        print("EVALUATION PROTOCOLS DIFFER -- refusing to compare.")
        for k, (b, a) in mismatch.items():
            print(f"  {k}: baseline={b!r}  ablation={a!r}")
        print("A protocol difference can manufacture a multi-dB 'improvement'.")
        print("Re-evaluate both models identically, or pass --allow-protocol-mismatch.")
        return 2

    # Pair strictly by filename: an index-based join would silently misalign if
    # either run skipped an image.
    shared = sorted(set(base_rows) & set(abl_rows))
    only_base = sorted(set(base_rows) - set(abl_rows))
    only_abl = sorted(set(abl_rows) - set(base_rows))
    if not shared:
        print("No filenames in common between the two runs.")
        return 2
    if only_base or only_abl:
        print(f"WARNING: {len(only_base)} images only in baseline, "
              f"{len(only_abl)} only in ablation; comparing the {len(shared)} shared.")

    report = {
        "baseline": {"label": args.baseline_label, "dir": args.baseline,
                     "experiment": base_sum.get("experiment"),
                     "checkpoint": base_sum.get("checkpoint"),
                     "epoch": base_sum.get("epoch")},
        "ablation": {"label": args.ablation_label, "dir": args.ablation,
                     "experiment": abl_sum.get("experiment"),
                     "checkpoint": abl_sum.get("checkpoint"),
                     "epoch": abl_sum.get("epoch")},
        "protocol": pb,
        "protocol_mismatch": mismatch or None,
        "n_paired_images": len(shared),
        "images_only_in_baseline": only_base,
        "images_only_in_ablation": only_abl,
        "delta_definition": f"delta = {args.ablation_label} - {args.baseline_label}",
        "metrics": {},
        "caveats": [
            "The 95% CI is over TEST IMAGES for one fixed pair of checkpoints. "
            "It is NOT a training-run variance estimate.",
            "Each arm is a SINGLE training run. Run-to-run (seed) variance is "
            "unmeasured, so a significant per-image difference may still be seed "
            "noise rather than an effect of the module.",
            "Validation and test are the same split in this repository, so both "
            "checkpoints were selected on the data they are scored on.",
        ],
    }

    print("=" * 78)
    print(f"PAIRED COMPARISON  ({len(shared)} images, "
          f"{pb['split']} split, {pb['crop_mode']} {pb['crop_size']}px)")
    print(f"delta = {args.ablation_label} - {args.baseline_label}")
    print("=" * 78)

    for metric, direction in METRIC_DIRECTION.items():
        pairs = [(base_rows[f].get(metric), abl_rows[f].get(metric)) for f in shared]
        pairs = [(b, a) for b, a in pairs if b is not None and a is not None]
        if not pairs:
            print(f"{metric}: not present in both runs, skipped")
            continue

        b = np.array([p[0] for p in pairs], dtype=np.float64)
        a = np.array([p[1] for p in pairs], dtype=np.float64)
        stats_d = analyse(a - b, direction, seed=args.seed)
        stats_d["baseline_mean"] = float(b.mean())
        stats_d["ablation_mean"] = float(a.mean())
        report["metrics"][metric] = stats_d

        arrow = "higher better" if direction > 0 else "LOWER better"
        print(f"\n{metric.upper()}  ({arrow})")
        print(f"  {args.baseline_label:<24} {b.mean():.4f}")
        print(f"  {args.ablation_label:<24} {a.mean():.4f}")
        print(f"  delta mean   {stats_d['mean']:+.4f}   median {stats_d['median']:+.4f}"
              f"   std {stats_d['std']:.4f}")
        print(f"  95% CI (t)   [{stats_d['ci95_mean'][0]:+.4f}, {stats_d['ci95_mean'][1]:+.4f}]"
              f"   bootstrap [{stats_d['ci95_bootstrap'][0]:+.4f}, "
              f"{stats_d['ci95_bootstrap'][1]:+.4f}]")
        print(f"  better on    {stats_d['improved_images']}/{stats_d['n_images']} images"
              f"   worse on {stats_d['worsened_images']}")
        if "paired_t_p" in stats_d:
            print(f"  paired t p={stats_d['paired_t_p']:.3e}   "
                  f"Wilcoxon p={stats_d.get('wilcoxon_p', float('nan')):.3e}   "
                  f"d={stats_d.get('cohens_d_paired', float('nan')):.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}")

    if args.markdown:
        os.makedirs(os.path.dirname(args.markdown) or ".", exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(write_markdown(report, args))
        print(f"wrote {args.markdown}")

    print("\nREMINDER: the CI is across test images for two fixed checkpoints.")
    print("It is NOT a training-run variance estimate; each arm is one run.")
    return 0


def write_markdown(report: dict, args) -> str:
    p = report["protocol"]
    L = [
        f"# Paired comparison — {report['ablation']['label']} vs {report['baseline']['label']}",
        "",
        f"**Images:** {report['n_paired_images']} paired by filename  ",
        f"**Protocol:** {p['split']} split, `{p['crop_mode']}` crop, {p['crop_size']} px  ",
        f"**Baseline checkpoint:** epoch {report['baseline']['epoch']}  ",
        f"**Ablation checkpoint:** epoch {report['ablation']['epoch']}  ",
        f"**Delta definition:** `{report['delta_definition']}`",
        "",
        "| Metric | " + report["baseline"]["label"] + " | " + report["ablation"]["label"]
        + " | Δ mean | Δ median | Δ std | 95% CI on Δ mean | better on |",
        "|---|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for m, s in report["metrics"].items():
        lo, hi = s["ci95_mean"]
        L.append(
            f"| {m.upper()} ({'↑' if s['direction'] == 'higher_better' else '↓'}) "
            f"| {s['baseline_mean']:.4f} | {s['ablation_mean']:.4f} "
            f"| {s['mean']:+.4f} | {s['median']:+.4f} | {s['std']:.4f} "
            f"| [{lo:+.4f}, {hi:+.4f}] "
            f"| {s['improved_images']}/{s['n_images']} |"
        )
    L += ["", "## How to read this", ""]
    for c in report["caveats"]:
        L.append(f"* {c}")
    L += ["",
          "`↑` higher is better, `↓` lower is better. A negative ΔLPIPS is an "
          "improvement.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
