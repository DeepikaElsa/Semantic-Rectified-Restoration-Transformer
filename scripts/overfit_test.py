"""
Phase 3 -- small overfit test.

Trains the Restormer baseline on a VERY SMALL fixed subset of GoPro until it
should be able to memorize it.

This is NOT a benchmark. It is a debugging experiment answering one question:
can this pipeline learn at all? A model that cannot overfit 16 images will never
learn 2103, and the bug must be found before spending hours on training.

Deterministic by construction: fixed sample indices, center crops, no
augmentation, fixed seed -- so a failure is reproducible.

Outputs to results/sanity_overfit/:
    metrics.csv, summary.json, curves.png
    images/sample<k>_{input,target,pred_initial,pred_final,comparison}.png

    python scripts/overfit_test.py
    python scripts/overfit_test.py --num-images 16 --epochs 40 --patch 128
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import build_dataset  # noqa: E402
from evaluate import compute_ssim  # noqa: E402
from losses.combined_loss import CombinedLoss  # noqa: E402
from models.srrt import SRRT  # noqa: E402
from train import build_optimizer, compute_psnr, model_output  # noqa: E402
from utils.ablation import ABLATIONS, apply_ablation  # noqa: E402
from utils.seed import collect_environment, set_seed  # noqa: E402


@torch.no_grad()
def evaluate_subset(model, loader, device, use_amp):
    """Mean L1 / PSNR / SSIM over the fixed subset, in eval mode."""
    model.eval()
    l1s, psnrs, ssims = [], [], []

    for batch in loader:
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model_output(model, inp)
        pred = pred.float().clamp(0, 1)

        l1s.append(float(torch.nn.functional.l1_loss(pred, tgt)))
        psnrs.append(compute_psnr(pred, tgt))
        for i in range(pred.shape[0]):
            ssims.append(compute_ssim(pred[i].cpu(), tgt[i].cpu()))

    model.train()
    return float(np.mean(l1s)), float(np.mean(psnrs)), float(np.mean(ssims))


@torch.no_grad()
def dump_predictions(model, loader, device, use_amp, out_dir, tag, limit=4):
    model.eval()
    saved = 0
    for batch in loader:
        inp = batch["input"].to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model_output(model, inp)
        pred = pred.float().clamp(0, 1)
        for i in range(pred.shape[0]):
            if saved >= limit:
                model.train()
                return
            save_image(pred[i], os.path.join(out_dir, f"sample{saved}_pred_{tag}.png"))
            saved += 1
    model.train()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/srrt_base.yaml")
    ap.add_argument("--num-images", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None, help="override optimizer.learning_rate")
    ap.add_argument("--out", default="results/sanity_overfit")
    ap.add_argument("--ablation", default="baseline_restormer", choices=sorted(ABLATIONS),
                    help="which ablation preset to overfit; default is the Phase 3 baseline")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = apply_ablation(cfg, args.ablation)
    if args.lr is not None:
        cfg["optimizer"]["learning_rate"] = args.lr

    set_seed(cfg["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["device"]["mixed_precision"]) and device.type == "cuda"

    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            float(cfg["device"]["max_memory_fraction"]), 0)
        torch.cuda.reset_peak_memory_stats(device)

    image_dir = os.path.join(args.out, "images")
    os.makedirs(image_dir, exist_ok=True)

    print("=" * 70)
    print("PHASE 3 -- SMALL OVERFIT TEST (debugging experiment, not a benchmark)")
    print("=" * 70)

    # Fixed subset: deterministic indices, center crop, no augmentation.
    full = build_dataset(cfg["dataset"].get("name", "gopro"), cfg["dataset"]["root"],
                         split="train", crop_size=args.patch,
                         crop_mode="center", augment=False)
    indices = list(range(args.num_images))
    subset = Subset(full, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=True, num_workers=0)
    eval_loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)

    model = SRRT.from_config(cfg).to(device)
    loss_fn = CombinedLoss.from_config(cfg).to(device)
    opt = build_optimizer(cfg, model)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    print(f"subset={args.num_images} images  patch={args.patch}  batch=1  "
          f"epochs={args.epochs}  lr={cfg['optimizer']['learning_rate']}")
    print(f"device={device.type}  AMP={use_amp}  params={sum(p.numel() for p in model.parameters()):,}")
    print("-" * 70)

    # Save inputs/targets and the BEFORE-training predictions.
    for k, idx in enumerate(indices[:4]):
        s = full[idx]
        save_image(s["input"], os.path.join(image_dir, f"sample{k}_input.png"))
        save_image(s["target"], os.path.join(image_dir, f"sample{k}_target.png"))
    dump_predictions(model, eval_loader, device, use_amp, image_dir, "initial")

    l1_0, psnr_0, ssim_0 = evaluate_subset(model, eval_loader, device, use_amp)
    print(f"epoch  0 (before training): L1={l1_0:.6f}  PSNR={psnr_0:.3f}dB  SSIM={ssim_0:.4f}")

    history = [{"epoch": 0, "train_loss": None, "l1": l1_0, "psnr": psnr_0, "ssim": ssim_0}]
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        running, n = 0.0, 0
        for batch in loader:
            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss, _ = loss_fn(model_output(model, inp), tgt)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.detach())
            n += 1

        train_loss = running / max(n, 1)
        l1, psnr, ssim = evaluate_subset(model, eval_loader, device, use_amp)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "l1": l1, "psnr": psnr, "ssim": ssim})

        if epoch % max(args.epochs // 10, 1) == 0 or epoch == 1:
            print(f"epoch {epoch:>3}: train_loss={train_loss:.6f}  "
                  f"L1={l1:.6f}  PSNR={psnr:.3f}dB  SSIM={ssim:.4f}")

    elapsed = time.time() - t0
    dump_predictions(model, eval_loader, device, use_amp, image_dir, "final")

    # Side-by-side comparison: input | initial pred | final pred | target
    from PIL import Image
    for k in range(min(4, args.num_images)):
        try:
            tiles = [Image.open(os.path.join(image_dir, f"sample{k}_{n}.png")).convert("RGB")
                     for n in ("input", "pred_initial", "pred_final", "target")]
        except FileNotFoundError:
            continue
        w, h = tiles[0].size
        strip = Image.new("RGB", (w * 4, h))
        for i, tile in enumerate(tiles):
            strip.paste(tile, (i * w, 0))
        strip.save(os.path.join(image_dir, f"sample{k}_comparison.png"))

    l1_f, psnr_f, ssim_f = history[-1]["l1"], history[-1]["psnr"], history[-1]["ssim"]
    peak_vram = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                 if device.type == "cuda" else 0.0)

    checks = {
        "train_loss_decreased": history[-1]["train_loss"] < history[1]["train_loss"],
        "l1_decreased": l1_f < l1_0,
        "psnr_increased": psnr_f > psnr_0,
        "ssim_increased": ssim_f > ssim_0,
        "psnr_gain_at_least_3db": (psnr_f - psnr_0) >= 3.0,
        "all_finite": all(np.isfinite([l1_f, psnr_f, ssim_f])),
    }

    print("-" * 70)
    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print("-" * 70)
    print(f"L1   {l1_0:.6f} -> {l1_f:.6f}")
    print(f"PSNR {psnr_0:.3f} -> {psnr_f:.3f} dB   (gain {psnr_f - psnr_0:+.3f} dB)")
    print(f"SSIM {ssim_0:.4f} -> {ssim_f:.4f}")
    print(f"time {elapsed:.1f}s   peak VRAM {peak_vram:.0f} MB")

    with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "l1", "psnr", "ssim"])
        w.writeheader()
        w.writerows(history)

    summary = {
        "purpose": "debugging experiment: verify the pipeline can learn. NOT a benchmark.",
        "config": "baseline_restormer", "num_images": args.num_images,
        "patch": args.patch, "epochs": args.epochs, "batch_size": 1,
        "augmentation": False, "crop_mode": "center",
        "learning_rate": cfg["optimizer"]["learning_rate"],
        "seed": cfg["experiment"]["seed"],
        "initial": {"l1": l1_0, "psnr": psnr_0, "ssim": ssim_0},
        "final": {"l1": l1_f, "psnr": psnr_f, "ssim": ssim_f},
        "checks": checks, "all_passed": all(checks.values()),
        "elapsed_s": elapsed, "peak_vram_mb": peak_vram,
        "params": sum(p.numel() for p in model.parameters()),
        "environment": collect_environment(),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ep = [h["epoch"] for h in history]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(ep, [h["l1"] for h in history]); ax[0].set_title("L1 (lower better)")
        ax[1].plot(ep, [h["psnr"] for h in history]); ax[1].set_title("PSNR dB (higher better)")
        ax[2].plot(ep, [h["ssim"] for h in history]); ax[2].set_title("SSIM (higher better)")
        for a in ax:
            a.set_xlabel("epoch"); a.grid(alpha=0.3)
        fig.suptitle(f"Overfit test on {args.num_images} fixed GoPro pairs "
                     f"(debugging experiment, not a benchmark)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "curves.png"), dpi=110)
        plt.close(fig)
    except Exception as exc:  # plotting is optional
        print(f"(plot skipped: {exc})")

    ok = all(checks.values())
    print("=" * 70)
    print("OVERFIT TEST: " + ("PASSED -- the pipeline can learn; proceed to baseline training"
                              if ok else "FAILED -- diagnose before training"))
    print("=" * 70)
    print(f"Written to {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
