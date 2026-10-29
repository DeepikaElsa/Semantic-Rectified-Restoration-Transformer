"""
Phase 2 -- Restormer baseline sanity test.

All SRRT modules disabled, batch size 1, small patch, few iterations, AMP on.
Gates a long training run: if any check here fails, do not proceed.

Verifies:
  1. forward produces finite values
  2. loss is finite
  3. backward produces finite gradients
  4. optimizer actually updates parameters
  5. loss decreases over the iterations
  6. PSNR is never NaN/Inf
  7. GPU VRAM stays within the configured cap
  8. CPU RAM does not explode

    python scripts/sanity_check.py
    python scripts/sanity_check.py --iters 20 --patch 128
"""

import argparse
import json
import os
import sys

import numpy as np
import psutil
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import create_dataloader  # noqa: E402
from losses.combined_loss import CombinedLoss  # noqa: E402
from models.srrt import SRRT  # noqa: E402
from train import build_optimizer, compute_psnr, model_output  # noqa: E402
from utils.ablation import ABLATIONS, apply_ablation  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/srrt_base.yaml")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--json", default="results/sanity_check/sanity.json")
    ap.add_argument("--ablation", default="baseline_restormer", choices=sorted(ABLATIONS),
                    help="which ablation preset to gate; the default is the Phase 2 baseline")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Phase 2 requirement: the model must reduce to plain Restormer. Later
    # ablations gate their own long run through the same checks, so the preset
    # is selectable rather than hardcoded.
    cfg = apply_ablation(cfg, args.ablation)
    set_seed(cfg["experiment"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["device"]["mixed_precision"]) and device.type == "cuda"

    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            float(cfg["device"]["max_memory_fraction"]), 0)
        torch.cuda.reset_peak_memory_stats(device)

    proc = psutil.Process()
    rss_start = proc.memory_info().rss / 1024 ** 2

    print("=" * 70)
    print("PHASE 2 -- RESTORMER BASELINE SANITY TEST")
    print("=" * 70)
    print(f"device={device.type}  AMP={use_amp}  batch={args.batch_size}  "
          f"patch={args.patch}  iters={args.iters}")

    model = SRRT.from_config(cfg).to(device)
    active = [k for k, v in model.module_summary().items()
              if isinstance(v, dict) and v.get("enabled") and k != "backbone"]
    print(f"active SRRT modules={active if active else 'NONE (pure Restormer)'}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    print("-" * 70)

    loss_fn = CombinedLoss.from_config(cfg).to(device)
    opt = build_optimizer(cfg, model)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    loader = create_dataloader(
        root=cfg["dataset"]["root"], name=cfg["dataset"].get("name", "gopro"),
        split="train", batch_size=args.batch_size, crop_size=args.patch,
        crop_mode="random", augment=False, shuffle=True, num_workers=0,
    )

    # One fixed batch: the point is to see the loss move on identical data, not
    # to average over the dataset.
    batch = next(iter(loader))
    inp = batch["input"].to(device)
    tgt = batch["target"].to(device)

    params_before = [p.detach().clone() for p in model.parameters()]

    history = []
    finite_forward = finite_loss = finite_grads = psnr_finite = True

    model.train()
    for it in range(1, args.iters + 1):
        opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model_output(model, inp)
            loss, comps = loss_fn(pred, tgt)

        finite_forward &= bool(torch.isfinite(pred).all())
        finite_loss &= bool(torch.isfinite(loss))

        scaler.scale(loss).backward()

        # Inspect gradients after unscaling, otherwise the loss-scale factor
        # makes the magnitudes meaningless.
        scaler.unscale_(opt)
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        finite_grads &= all(bool(torch.isfinite(g).all()) for g in grads)
        gnorm = float(torch.norm(torch.stack([g.norm() for g in grads])))

        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            psnr = compute_psnr(pred.detach().float().clamp(0, 1), tgt)
        psnr_finite &= bool(np.isfinite(psnr))

        vram = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                if device.type == "cuda" else 0.0)
        rss = proc.memory_info().rss / 1024 ** 2

        history.append({"iter": it, "loss": float(loss.detach()),
                        "l1": float(comps["l1"].detach()), "psnr": psnr,
                        "grad_norm": gnorm, "vram_mb": vram, "rss_mb": rss})

        if it == 1 or it == args.iters or it % max(args.iters // 4, 1) == 0:
            print(f"Iteration {it}:")
            print(f"  Loss:     {float(loss.detach()):.6f}   (L1 {float(comps['l1'].detach()):.6f})")
            print(f"  PSNR:     {psnr:.3f} dB")
            print(f"  GradNorm: {gnorm:.4f}")
            print(f"  GPU VRAM: {vram:.0f} MB")
            print(f"  CPU RSS:  {rss:.0f} MB")

    print("-" * 70)

    changed = sum(1 for a, b in zip(params_before, model.parameters())
                  if not torch.equal(a, b.detach()))
    first, last = history[0]["loss"], history[-1]["loss"]
    peak_vram = max(h["vram_mb"] for h in history)
    peak_rss = max(h["rss_mb"] for h in history)
    cap = (float(cfg["device"]["max_memory_fraction"])
           * torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
           if device.type == "cuda" else float("inf"))

    checks = {
        "1_forward_finite": finite_forward,
        "2_loss_finite": finite_loss,
        "3_gradients_finite": finite_grads,
        "4_optimizer_updates_params": changed > 0,
        "5_loss_decreased": last < first,
        "6_psnr_finite": psnr_finite,
        "7_vram_within_cap": peak_vram < cap,
        "8_cpu_ram_stable": (peak_rss - rss_start) < 8192,
    }

    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    print("-" * 70)
    print(f"loss {first:.6f} -> {last:.6f}  (delta {last - first:+.6f})")
    print(f"PSNR {history[0]['psnr']:.3f} -> {history[-1]['psnr']:.3f} dB")
    print(f"params changed by optimizer: {changed}/{len(params_before)}")
    print(f"peak VRAM {peak_vram:.0f} MB (cap {cap:.0f} MB)")
    print(f"CPU RSS {rss_start:.0f} -> {peak_rss:.0f} MB (delta {peak_rss - rss_start:+.0f} MB)")

    all_ok = all(checks.values())
    print("=" * 70)
    print("SANITY TEST: " + ("PASSED -- safe to proceed to the overfit test"
                             if all_ok else "FAILED -- do NOT start training"))
    print("=" * 70)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"config": "baseline_restormer", "device": device.type,
                       "amp": use_amp, "batch_size": args.batch_size,
                       "patch": args.patch, "iters": args.iters,
                       "checks": checks, "history": history,
                       "peak_vram_mb": peak_vram, "peak_rss_mb": peak_rss,
                       "all_passed": all_ok}, f, indent=2)
        print(f"Written to {args.json}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
