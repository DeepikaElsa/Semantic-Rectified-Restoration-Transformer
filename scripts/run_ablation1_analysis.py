"""
Post-training analysis pipeline for Ablation 1 (Restormer + SeeCoder).

Runs, in order, once the 60-epoch `seecoder` run has finished:

     1  final evaluation            evaluate.py on the best-PSNR checkpoint
     2  convergence analysis        analyze_convergence.py
     3  checkpoint selection        reported from the checkpoint + log
     4  quantitative comparison     compare_paired_metrics.py, tables
     5  qualitative comparison      generate_qualitative_grid.py (x2)
     6  error maps                  visualize_error_maps.py
     7  frequency analysis          visualize_frequency.py
     8  feature visualization       visualize_features.py
     9  efficiency comparison       efficiency_table.py
    10  documentation reminder      printed; the write-up stays a human decision

    python scripts/run_ablation1_analysis.py
    python scripts/run_ablation1_analysis.py --dry-run     # print the plan only

GUARD
-----
Refuses to start while training is unfinished. Evaluating a partially trained
checkpoint and filing it as "Ablation 1" is the single easiest way to corrupt the
experimental record, so the epoch count is checked against --expect-epochs and
the run aborts unless --force is given.

It NEVER launches training and NEVER starts the next ablation. Deciding whether
to run DiffAE is a research decision that comes after these results are read.

The qualitative, error, frequency and feature stages all use the SAME
deterministic image IDs (--seed / --num-samples), chosen by
`utils.research.select_image_ids`, which sees no model and no metric. Sample
selection therefore cannot be steered toward images that flatter either model.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.research import read_train_log, resolve_train_log, to_float  # noqa: E402

PY = sys.executable

BASE_EXP = "baseline_restormer"
ABL_EXP = "seecoder"
BASE_LABEL = "Restormer"
ABL_LABEL = "Restormer + SeeCoder"
BASE_RESULTS = "results/restormer_baseline"
ABL_RESULTS = "results/ablation_seecoder"
BASE_CKPT = "checkpoints/baseline_restormer/best_psnr.pth"
ABL_CKPT = "checkpoints/seecoder/best_psnr.pth"
FIG = "research/figures"


def run(step: str, cmd: list[str], dry: bool, cont: bool) -> bool:
    print("\n" + "=" * 78)
    print(f"STEP {step}")
    print("$ " + " ".join(cmd))
    print("=" * 78)
    if dry:
        return True
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"\n!! step {step} exited {rc}")
        if not cont:
            print("Stopping. Re-run with --continue-on-error to push past this.")
        return False
    return True


def guard(expect: int, force: bool) -> bool:
    try:
        log = resolve_train_log(ABL_EXP)
    except FileNotFoundError:
        print(f"No training log for {ABL_EXP!r}; nothing to analyse.")
        return False

    rows = read_train_log(log)
    done = len(rows)
    last = int(to_float(rows[-1]["epoch"])) if rows else -1
    print(f"{ABL_EXP}: {done} epochs logged (last epoch index {last})")

    if done < expect:
        print(f"\nTRAINING IS NOT FINISHED: {done}/{expect} epochs.")
        print("Evaluating a partially trained checkpoint would misrepresent")
        print("Ablation 1. Wait for the run, or pass --force if you have a")
        print("specific reason to analyse an incomplete run.")
        return bool(force)

    if not os.path.isfile(ABL_CKPT):
        print(f"\nMissing {ABL_CKPT}.")
        return bool(force)
    return True


def report(dry: bool) -> None:
    """Print the final Ablation 1 summary from measured artifacts only."""
    if dry:
        return
    paired = "research/tables/ablation_seecoder_paired.json"
    if not os.path.isfile(paired):
        print("\n(no paired comparison written; cannot print the summary)")
        return
    with open(paired) as f:
        rep = json.load(f)

    m = rep["metrics"]
    print("\n" + "=" * 78)
    print("ABLATION 1 RESULT")
    print("=" * 78)
    print(f"Protocol: {rep['protocol']['split']} split, {rep['n_paired_images']} images, "
          f"{rep['protocol']['crop_mode']} crop {rep['protocol']['crop_size']}px")
    print(f"\n{rep['baseline']['label']}  (epoch {rep['baseline']['epoch']}):")
    for k in ("psnr", "ssim", "lpips"):
        if k in m:
            print(f"  {k.upper():<6} = {m[k]['baseline_mean']:.4f}")
    print(f"\n{rep['ablation']['label']}  (epoch {rep['ablation']['epoch']}):")
    for k in ("psnr", "ssim", "lpips"):
        if k in m:
            print(f"  {k.upper():<6} = {m[k]['ablation_mean']:.4f}")
    print("\nDeltas (ablation - baseline):")
    for k in ("psnr", "ssim", "lpips"):
        if k in m:
            lo, hi = m[k]["ci95_mean"]
            print(f"  d{k.upper():<5} = {m[k]['mean']:+.4f}   "
                  f"95% CI [{lo:+.4f}, {hi:+.4f}]   "
                  f"better on {m[k]['improved_images']}/{m[k]['n_images']}")

    print("\n" + "-" * 78)
    print(verdict(m))
    print("-" * 78)
    print("\nRun-to-run variance is UNMEASURED (one training run per arm), so the")
    print("confidence intervals above bound sampling over IMAGES, not over seeds.")
    print("\nNOT starting DiffAE. The next experiment is a separate decision.")


def verdict(m: dict) -> str:
    """
    Mechanical reading of the evidence. Stated as a rule so it does not depend
    on which way the numbers happen to look.

    IMPROVEMENT   every available metric moves the beneficial way AND its 95% CI
                  excludes zero AND |dPSNR| >= 0.1 dB
    DEGRADATION   same, in the harmful direction
    INCONCLUSIVE  anything else -- including a significant but tiny effect, since
                  a single seed per arm cannot separate it from run-to-run noise
    """
    if not m:
        return "VERDICT: INCONCLUSIVE (no metrics available)"

    def good(metric: str) -> int:
        s = m[metric]
        sign = 1 if s["direction"] == "higher_better" else -1
        if not s["ci_excludes_zero"]:
            return 0
        return 1 if s["mean"] * sign > 0 else -1

    signs = {k: good(k) for k in m}
    dpsnr = abs(m["psnr"]["mean"]) if "psnr" in m else 0.0

    if all(v == 1 for v in signs.values()) and dpsnr >= 0.1:
        head = "VERDICT: YES -- evidence of improvement (single seed)"
    elif all(v == -1 for v in signs.values()) and dpsnr >= 0.1:
        head = "VERDICT: NO -- evidence of degradation (single seed)"
    else:
        head = "VERDICT: INCONCLUSIVE"

    detail = ", ".join(
        f"{k.upper()}: " + {1: "better", -1: "worse", 0: "CI includes 0"}[v]
        for k, v in signs.items())
    return f"{head}\n  {detail}\n  |dPSNR| = {dpsnr:.4f} dB (threshold 0.1 dB)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--expect-epochs", type=int, default=60)
    ap.add_argument("--force", action="store_true",
                    help="analyse even if training is unfinished (records nothing extra)")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--num-samples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-eval", action="store_true",
                    help="reuse an existing results/ablation_seecoder evaluation")
    args = ap.parse_args()

    # --dry-run executes nothing, so it may inspect the plan mid-training.
    if not guard(args.expect_epochs, args.force or args.dry_run):
        return 1

    models = [f"{BASE_LABEL}={BASE_CKPT}", f"{ABL_LABEL}={ABL_CKPT}"]
    model_args: list[str] = []
    for m in models:
        model_args += ["--model", m]
    common = ["--num_samples", str(args.num_samples), "--seed", str(args.seed),
              "--split", "test"]

    steps: list[tuple[str, list[str]]] = []

    if not args.skip_eval:
        steps.append(("1  final evaluation",
                      [PY, "evaluate.py", "--checkpoint", ABL_CKPT, "--split", "test",
                       "--out-dir", ABL_RESULTS, "--save-images", "8"]))

    steps += [
        ("2  convergence analysis",
         [PY, "scripts/analyze_convergence.py", "--experiment", ABL_EXP,
          "--json", f"{ABL_RESULTS}/convergence.json"]),

        ("4a quantitative -- paired per-image comparison",
         [PY, "scripts/compare_paired_metrics.py",
          "--baseline", BASE_RESULTS, "--ablation", ABL_RESULTS,
          "--baseline-label", BASE_LABEL, "--ablation-label", ABL_LABEL]),

        ("4b quantitative -- ablation table",
         [PY, "scripts/make_comparison_table.py",
          "--markdown", "research/tables/ablation_main.md"]),

        ("4c quantitative -- ablation ladder figure",
         [PY, "scripts/compare_ablation_results.py",
          "--out-dir", FIG, "--stem", "ablation_comparison", "--errorbars"]),

        ("4d quantitative -- convergence comparison figure",
         [PY, "scripts/plot_convergence_comparison.py",
          "--run", f"{BASE_LABEL}={BASE_EXP}", "--run", f"{ABL_LABEL}={ABL_EXP}",
          "--out-dir", FIG, "--stem", "ablation_seecoder_convergence"]),

        ("5a qualitative -- Input | GT | Restormer | +SeeCoder",
         [PY, "scripts/generate_qualitative_grid.py", *model_args, *common,
          "--output_dir", FIG, "--stem", "ablation_seecoder_qualitative"]),

        ("5b qualitative -- ... | Error",
         [PY, "scripts/generate_qualitative_grid.py", *model_args, *common,
          "--output_dir", f"{FIG}/qualitative_seecoder", "--with-error",
          "--stem", "ablation_seecoder_qualitative_error"]),

        ("6  error maps",
         [PY, "scripts/visualize_error_maps.py", *model_args, *common,
          "--output_dir", FIG, "--modes", "abs_mean,squared_mean",
          "--shared-scale-across-images"]),

        ("7  frequency analysis",
         [PY, "scripts/visualize_frequency.py", *model_args, *common,
          "--output_dir", FIG]),

        ("8  feature visualization",
         [PY, "scripts/visualize_features.py", "--checkpoint", ABL_CKPT,
          "--seed", str(args.seed), "--split", "test",
          "--output_dir", f"{FIG}/features_seecoder"]),

        ("9  efficiency comparison",
         [PY, "scripts/efficiency_table.py",
          "--run", f"{BASE_LABEL}={BASE_EXP}={BASE_RESULTS}",
          "--run", f"{ABL_LABEL}={ABL_EXP}={ABL_RESULTS}"]),
    ]

    for name, cmd in steps:
        if not run(name, cmd, args.dry_run, args.continue_on_error):
            if not args.continue_on_error:
                return 1

    # Step 5b writes the error contact sheet into a subdirectory; surface it at
    # the documented path too.
    if not args.dry_run:
        import shutil
        for ext in ("png", "pdf", "json"):
            src = f"{FIG}/qualitative_seecoder/ablation_seecoder_qualitative_error.{ext}"
            if os.path.isfile(src):
                shutil.copy2(src, f"{FIG}/ablation_seecoder_qualitative_error.{ext}")

    report(args.dry_run)

    print("\nSTEP 10  documentation update -- NOT automated.")
    print("  Update research_notes/ABLATION1_SEECODER.md and VALIDATION_STATUS.md")
    print("  with the measured numbers and the verdict above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
