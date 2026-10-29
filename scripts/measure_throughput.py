"""
Measure steady-state training throughput and project training cost.

Answers the practical planning question: how long does one epoch actually take,
and is a 200-epoch run feasible on this machine?

MUST stay under `if __name__ == "__main__":` -- on Windows, DataLoader workers
use spawn and re-import the parent module, so an unguarded script recursively
spawns workers and hangs at 0% GPU utilization.

    python scripts/measure_throughput.py
    python scripts/measure_throughput.py --workers 0 --measure 30
"""

import argparse
import json
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import dataloader_from_config  # noqa: E402
from losses.combined_loss import CombinedLoss  # noqa: E402
from models.srrt import SRRT  # noqa: E402
from train import build_optimizer, model_output  # noqa: E402
from utils.ablation import apply_ablation  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/srrt_base.yaml")
    ap.add_argument("--ablation", default="baseline_restormer")
    ap.add_argument("--workers", type=int, default=None, help="override num_workers")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--measure", type=int, default=40)
    ap.add_argument("--val-images", type=int, default=15)
    ap.add_argument("--json", default="results/throughput.json")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = apply_ablation(cfg, args.ablation)
    if args.workers is not None:
        cfg["dataset"]["train"]["num_workers"] = args.workers
        cfg["dataset"]["val"]["num_workers"] = args.workers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["device"]["mixed_precision"]) and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            float(cfg["device"]["max_memory_fraction"]), 0)
        torch.backends.cudnn.benchmark = True

    t_setup = time.perf_counter()
    model = SRRT.from_config(cfg).to(device)
    loss_fn = CombinedLoss.from_config(cfg).to(device)
    opt = build_optimizer(cfg, model)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    setup_s = time.perf_counter() - t_setup

    tcfg = cfg["dataset"]["train"]
    t_dl = time.perf_counter()
    loader = dataloader_from_config(cfg, "train")
    it = iter(loader)
    first = next(it)          # includes worker spawn on Windows
    dl_start_s = time.perf_counter() - t_dl

    n_iters = len(loader)
    print("=" * 70)
    print("THROUGHPUT MEASUREMENT")
    print("=" * 70)
    print(f"ablation={args.ablation}  batch={tcfg['batch_size']}  crop={tcfg['crop_size']}  "
          f"workers={tcfg['num_workers']}  AMP={use_amp}")
    print(f"iters/epoch={n_iters}  model+loss setup={setup_s:.1f}s  "
          f"dataloader first-batch={dl_start_s:.1f}s")
    print("-" * 70)

    model.train()
    t0 = None
    for i in range(args.warmup + args.measure):
        batch = first if i == 0 else next(it)
        inp = batch["input"].to(device, non_blocking=True)
        tgt = batch["target"].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss, _ = loss_fn(model_output(model, inp), tgt)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if i == args.warmup - 1:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    ips = args.measure / dt
    ms = dt / args.measure * 1000
    epoch_min = n_iters / ips / 60
    print(f"TRAIN steady-state: {ips:.3f} it/s  ({ms:.0f} ms/iter)")
    print(f"  train-only epoch: {epoch_min:.1f} min")

    # Validation cost
    vcfg = cfg["dataset"]["val"]
    vl = dataloader_from_config(cfg, "val")
    n_val = len(vl.dataset)
    model.eval()
    vit = iter(vl)
    with torch.no_grad():
        # Warm up: the first batch pays Windows DataLoader worker spawn, which
        # would otherwise be amortized into the per-image number and inflate it
        # by ~1s/image over a short sample.
        model_output(model, next(vit)["input"].to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    tv = time.perf_counter()
    done = 0
    with torch.no_grad():
        for i, b in enumerate(vit):
            if i >= args.val_images:
                break
            model_output(model, b["input"].to(device))
            done += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    v_per = (time.perf_counter() - tv) / max(done, 1)
    val_min = v_per * n_val / 60
    print(f"VAL: {v_per * 1000:.0f} ms/image at {vcfg['crop_size']}px "
          f"-> full val set ({n_val} imgs) = {val_min:.1f} min")

    total_epoch_min = epoch_min + val_min
    print("-" * 70)
    print(f"epoch total (train + full val): {total_epoch_min:.1f} min")
    for e in (1, 5, 10, 20, 50, 200):
        h = total_epoch_min * e / 60
        print(f"  {e:>3} epochs: {h:>6.1f} h" + ("   <- config default" if e == 200 else ""))
    print("=" * 70)

    peak = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
            if device.type == "cuda" else 0.0)
    out = {
        "ablation": args.ablation, "batch_size": tcfg["batch_size"],
        "crop_size": tcfg["crop_size"], "num_workers": tcfg["num_workers"],
        "amp": use_amp, "iters_per_epoch": n_iters,
        "train_it_per_s": ips, "train_ms_per_iter": ms,
        "train_epoch_min": epoch_min,
        "val_ms_per_image": v_per * 1000, "val_images": n_val,
        "val_full_min": val_min, "epoch_total_min": total_epoch_min,
        "projected_hours": {str(e): total_epoch_min * e / 60
                            for e in (1, 5, 10, 20, 50, 200)},
        "setup_s": setup_s, "dataloader_first_batch_s": dl_start_s,
        "peak_vram_mb": peak,
    }
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
