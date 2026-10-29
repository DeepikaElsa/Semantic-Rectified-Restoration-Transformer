"""
Train + evaluate SRRT for one or all ablation stages.

    python scripts/run_srrt.py                       # full_srrt
    python scripts/run_srrt.py --ablation seecoder_diffae_lcr
    python scripts/run_srrt.py --all --max-steps 20  # sweep every stage (smoke)

`--all` runs stages A..F in order, which is the sequence needed to tell whether
LCR actually contributes. Comparison table: scripts/make_comparison_table.py
"""

import argparse
import subprocess
import sys

from utils.ablation import ABLATIONS


def run_one(name: str, config: str, max_steps: int | None, skip_eval: bool) -> None:
    train_cmd = [sys.executable, "train.py", "--config", config, "--ablation", name]
    if max_steps:
        train_cmd += ["--max-steps", str(max_steps), "--max-val-batches", "5"]

    print(f"\n=== [{name}] train ===\n$ " + " ".join(train_cmd))
    if subprocess.run(train_cmd).returncode != 0:
        raise SystemExit(f"Training failed for {name}.")

    if skip_eval:
        return

    eval_cmd = [sys.executable, "evaluate.py",
                "--checkpoint", f"checkpoints/{name}/best_psnr.pth"]
    if max_steps:
        eval_cmd += ["--limit", "20"]

    print(f"\n=== [{name}] evaluate ===\n$ " + " ".join(eval_cmd))
    if subprocess.run(eval_cmd).returncode != 0:
        raise SystemExit(f"Evaluation failed for {name}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/srrt_base.yaml")
    parser.add_argument("--ablation", default="full_srrt", choices=sorted(ABLATIONS))
    parser.add_argument("--all", action="store_true", help="run every ablation stage in order")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    stages = list(ABLATIONS) if args.all else [args.ablation]
    for name in stages:
        run_one(name, args.config, args.max_steps, args.skip_eval)


if __name__ == "__main__":
    main()
