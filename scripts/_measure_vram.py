"""
Measure peak training VRAM in a clean process.

Run as a subprocess (by scripts/audit_pipeline.py) so the measurement is not
contaminated by anything the caller has resident on the GPU. Only the model,
loss, optimizer and one batch are alive here -- exactly what real training holds.

Prints a single JSON line.

    python scripts/_measure_vram.py [--ablation NAME] [--steps N]
"""

import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import dataloader_from_config  # noqa: E402
from losses.combined_loss import CombinedLoss  # noqa: E402
from models.srrt import SRRT  # noqa: E402
from train import build_optimizer  # noqa: E402
from utils.ablation import apply_ablation  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/srrt_base.yaml")
    ap.add_argument("--ablation", default="baseline_restormer")
    ap.add_argument("--steps", type=int, default=2)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = apply_ablation(cfg, args.ablation)
    cfg["dataset"]["train"]["num_workers"] = 0

    if not torch.cuda.is_available():
        print(json.dumps({"cuda": False}))
        return 0

    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(
        float(cfg["device"].get("max_memory_fraction", 0.85)), 0
    )

    use_amp = bool(cfg["device"]["mixed_precision"])
    model = SRRT.from_config(cfg).to(device)
    loss_fn = CombinedLoss.from_config(cfg).to(device)
    opt = build_optimizer(cfg, model)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    batch = next(iter(dataloader_from_config(cfg, "train")))
    inp = batch["input"].to(device)
    tgt = batch["target"].to(device)

    torch.cuda.reset_peak_memory_stats(device)

    try:
        # Several steps so AdamW moment buffers (allocated lazily on step 1)
        # are included in the peak.
        for _ in range(args.steps):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                out = model(inp, target=tgt)
                loss, _ = loss_fn(out["output"], tgt,
                                  flow_loss=out.get("flow_loss"),
                                  lcr_loss=out.get("lcr_loss"))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        oom = False
    except torch.cuda.OutOfMemoryError:
        oom = True

    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
    print(json.dumps({
        "cuda": True,
        "ablation": args.ablation,
        "oom": oom,
        "peak_mb": round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 1),
        "reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1024 ** 2, 1),
        "total_vram_mb": round(total, 1),
        "cap_mb": round(float(cfg["device"]["max_memory_fraction"]) * total, 1),
        "batch_size": cfg["dataset"]["train"]["batch_size"],
        "crop_size": cfg["dataset"]["train"]["crop_size"],
        "amp": use_amp,
        "params": sum(p.numel() for p in model.parameters()),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
