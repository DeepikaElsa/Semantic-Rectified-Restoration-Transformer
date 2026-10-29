"""
Evaluation entry point.

Computes PSNR, SSIM and LPIPS over a dataset split, saves per-image metrics to
CSV, a summary JSON, and qualitative image triptychs.

Metric conventions
------------------
    PSNR = 10 * log10(MAX_I^2 / MSE),  MAX_I = 1 for [0,1] images   (higher better)
    SSIM : scikit-image, channel_axis=2, data_range=1              (higher better)
    LPIPS: lpips package, inputs rescaled to [-1,1]                (LOWER better)

LPIPS is deliberately NOT the training objective (training uses a VGG19
perceptual loss) so that this number stays an independent measure. See
docs/design_decisions.md.

Usage:
    python evaluate.py --checkpoint checkpoints/baseline_restormer/best_psnr.pth
    python evaluate.py --checkpoint <ckpt> --split test --limit 100
    python evaluate.py --checkpoint <ckpt> --full-images
"""

import argparse
import csv
import json
import os
import time
from typing import Any

import numpy as np
import torch
import yaml
from skimage.metrics import structural_similarity
from torchvision.utils import save_image

from data.datasets import create_dataloader
from models.srrt import SRRT
from utils.seed import collect_environment, set_seed


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * float(np.log10((max_val ** 2) / mse))


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """SSIM on a single (3,H,W) image pair in [0,1]."""
    p = pred.detach().permute(1, 2, 0).cpu().numpy().astype(np.float64)
    t = target.detach().permute(1, 2, 0).cpu().numpy().astype(np.float64)
    return float(structural_similarity(t, p, channel_axis=2, data_range=1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an SRRT / Restormer checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None,
                        help="override config; by default the config stored in the checkpoint is used")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N images")
    parser.add_argument("--full-images", action="store_true",
                        help="evaluate on full padded images instead of the configured crop")
    parser.add_argument("--save-images", type=int, default=8,
                        help="how many qualitative triptychs to write (0 disables)")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only=False: our checkpoints intentionally embed the config dict
    # and environment metadata (spec section 24), not just tensors.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    elif "config" in ckpt:
        cfg = ckpt["config"]
        print("Using the config stored in the checkpoint.")
    else:
        raise SystemExit("Checkpoint has no stored config; pass --config explicitly.")

    set_seed(cfg["experiment"]["seed"])

    run_name = cfg["experiment"]["name"]
    out_dir = args.out_dir or os.path.join(cfg["experiment"].get("results_dir", "results"), run_name)
    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            float(cfg["device"].get("max_memory_fraction", 0.85)), device.index or 0
        )

    val_cfg = cfg["dataset"]["val"]
    crop_mode = "none" if args.full_images else val_cfg.get("crop_mode", "center")

    loader = create_dataloader(
        root=cfg["dataset"]["root"],
        name=cfg["dataset"].get("name", "gopro"),
        split=args.split,
        batch_size=1,                       # per-image metrics require batch 1
        crop_size=val_cfg["crop_size"],
        crop_mode=crop_mode,
        augment=False,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model = SRRT.from_config(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    summary_modules = model.module_summary()
    n_params = summary_modules["total"]

    lpips_fn = None
    if "LPIPS" in cfg["evaluation"].get("metrics", []):
        import lpips
        lpips_fn = lpips.LPIPS(net=cfg["evaluation"].get("lpips_net", "alex")).to(device)
        lpips_fn.eval()

    rows: list[dict[str, Any]] = []
    n_saved = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    total_infer_time = 0.0

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if args.limit is not None and idx >= args.limit:
                break

            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = model(inp)["output"].clamp(0, 1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_time = time.perf_counter() - t0
            total_infer_time += infer_time

            row: dict[str, Any] = {
                "filename": batch["filename"][0],
                "psnr": compute_psnr(pred, tgt),
                "ssim": compute_ssim(pred[0], tgt[0]),
                "inference_s": infer_time,
            }

            if lpips_fn is not None:
                row["lpips"] = float(lpips_fn(pred * 2 - 1, tgt * 2 - 1).item())

            rows.append(row)

            if n_saved < args.save_images:
                stem = os.path.splitext(batch["filename"][0])[0]
                save_image(torch.cat([inp, pred, tgt], dim=0),
                           os.path.join(image_dir, f"{stem}_comparison.png"), nrow=3)
                save_image(inp, os.path.join(image_dir, f"{stem}_input.png"))
                save_image(pred, os.path.join(image_dir, f"{stem}_restored.png"))
                save_image(tgt, os.path.join(image_dir, f"{stem}_target.png"))
                n_saved += 1

            if (idx + 1) % 50 == 0:
                print(f"  {idx + 1} images... running PSNR "
                      f"{np.mean([r['psnr'] for r in rows]):.2f}dB")

    if not rows:
        raise SystemExit("No images were evaluated.")

    peak_vram = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                 if device.type == "cuda" else 0.0)

    def stats(key: str) -> dict[str, float]:
        values = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    metric_stats = {"psnr": stats("psnr"), "ssim": stats("ssim")}
    if lpips_fn is not None:
        metric_stats["lpips"] = stats("lpips")

    by_psnr = sorted(rows, key=lambda r: r["psnr"])
    summary = {
        "experiment": run_name,
        "checkpoint": args.checkpoint,
        "epoch": ckpt.get("epoch"),
        "split": args.split,
        "crop_mode": crop_mode,
        "crop_size": None if crop_mode == "none" else val_cfg["crop_size"],
        "num_images": len(rows),
        "metrics": metric_stats,
        "parameters": n_params,
        "module_summary": summary_modules,
        "mean_inference_s": total_infer_time / len(rows),
        "peak_vram_mb": peak_vram,
        "worst_5_by_psnr": [r["filename"] for r in by_psnr[:5]],
        "best_5_by_psnr": [r["filename"] for r in by_psnr[-5:]],
        "environment": collect_environment(),
        "metric_directions": {"psnr": "higher_better", "ssim": "higher_better",
                              "lpips": "lower_better"},
    }

    fieldnames = list(rows[0].keys())
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print("\n" + "=" * 62)
    print(f"Experiment      : {run_name}  ({len(rows)} images, {args.split})")
    print(f"Evaluated at    : crop_mode={crop_mode}"
          + ("" if crop_mode == "none" else f", crop_size={val_cfg['crop_size']}"))
    print(f"PSNR  (higher)  : {metric_stats['psnr']['mean']:.3f} +/- {metric_stats['psnr']['std']:.3f} dB")
    print(f"SSIM  (higher)  : {metric_stats['ssim']['mean']:.4f} +/- {metric_stats['ssim']['std']:.4f}")
    if lpips_fn is not None:
        print(f"LPIPS (lower)   : {metric_stats['lpips']['mean']:.4f} +/- {metric_stats['lpips']['std']:.4f}")
    print(f"Parameters      : {n_params:,}")
    print(f"Inference/image : {summary['mean_inference_s'] * 1000:.1f} ms")
    print(f"Peak VRAM       : {peak_vram:.0f} MB")
    print(f"Written to      : {out_dir}")
    print("=" * 62)


if __name__ == "__main__":
    main()
