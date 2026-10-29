"""
Train + evaluate the pure Restormer baseline (ablation stage A).

Exists so the baseline can be reproduced with one command, independently of any
SRRT module, which is what makes the ablation comparison fair.

    python scripts/run_baseline.py --epochs 5
"""

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/srrt_base.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    train_cmd = [sys.executable, "train.py", "--config", args.config,
                 "--ablation", "baseline_restormer"]
    if args.max_steps:
        train_cmd += ["--max-steps", str(args.max_steps)]

    if args.epochs is not None:
        print(f"NOTE: set training.epochs to {args.epochs} in {args.config} "
              f"(kept in YAML so the run stays reproducible from the config alone).")

    print("$ " + " ".join(train_cmd))
    if subprocess.run(train_cmd).returncode != 0:
        raise SystemExit("Training failed.")

    if args.skip_eval:
        return

    eval_cmd = [sys.executable, "evaluate.py",
                "--checkpoint", "checkpoints/baseline_restormer/best_psnr.pth"]
    print("$ " + " ".join(eval_cmd))
    if subprocess.run(eval_cmd).returncode != 0:
        raise SystemExit("Evaluation failed.")


if __name__ == "__main__":
    main()
