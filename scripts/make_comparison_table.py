"""
Collect results/<experiment>/summary.json files into a comparison table.

Reports ONLY what evaluate.py actually measured. Missing experiments are shown
as "not run" -- no value is ever interpolated, guessed, or carried over.

    python scripts/make_comparison_table.py
    python scripts/make_comparison_table.py --markdown results/comparison.md
"""

import argparse
import json
import os
import sys

# Run as `python scripts/make_comparison_table.py` from the repo root, as the
# docstring documents: the repo root is not on sys.path in that case.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ablation import ABLATIONS  # noqa: E402
from utils.research import find_summary  # noqa: E402

HEADER = ["Method", "PSNR (dB)", "SSIM", "LPIPS", "Params", "Epoch"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--markdown", default=None, help="also write a markdown table here")
    args = parser.parse_args()

    rows = []
    for name in ABLATIONS:
        # Resolve by alias / recorded experiment name, not just directory name.
        path = find_summary(args.results_dir, name)
        if path is None:
            rows.append([name, "not run", "not run", "not run", "-", "-"])
            continue

        with open(path, "r") as f:
            s = json.load(f)

        m = s["metrics"]
        lpips = f"{m['lpips']['mean']:.4f}" if "lpips" in m else "n/a"
        rows.append([
            name,
            f"{m['psnr']['mean']:.3f}",
            f"{m['ssim']['mean']:.4f}",
            lpips,
            f"{s['parameters']:,}",
            str(s.get("epoch", "-")),
        ])

    widths = [max(len(HEADER[i]), max(len(r[i]) for r in rows)) for i in range(len(HEADER))]

    def fmt(cells):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

    print("\nPSNR/SSIM: higher is better. LPIPS: lower is better.\n")
    print(fmt(HEADER))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt(r))

    if all(r[1] == "not run" for r in rows):
        print("\nNo experiments have been evaluated yet. Run scripts/run_srrt.py --all first.")

    if args.markdown:
        os.makedirs(os.path.dirname(args.markdown) or ".", exist_ok=True)
        with open(args.markdown, "w") as f:
            f.write("# SRRT ablation comparison\n\n")
            f.write("PSNR/SSIM: higher is better. LPIPS: lower is better.\n\n")
            f.write("| " + " | ".join(HEADER) + " |\n")
            f.write("|" + "|".join(["---"] * len(HEADER)) + "|\n")
            for r in rows:
                f.write("| " + " | ".join(r) + " |\n")
        print(f"\nMarkdown written to {args.markdown}")


if __name__ == "__main__":
    main()
