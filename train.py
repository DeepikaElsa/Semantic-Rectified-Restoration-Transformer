"""
Training entry point for SRRT (and the Restormer baseline).

The model is always built as `models.srrt.SRRT`. With every module disabled it is
numerically identical to the bare Restormer backbone (asserted in
tests/test_srrt.py), so the same loop serves every ablation stage without any
branching.

Windows note: DataLoader workers use 'spawn', so everything runs under
`if __name__ == "__main__":`.

Usage:
    python train.py --config configs/srrt_base.yaml
    python train.py --ablation full_srrt
    python train.py --resume checkpoints/latest.pth
    python train.py --max-steps 20 --max-val-batches 5     # smoke test
"""

import argparse
import csv
import json
import os
import time
from typing import Any

import psutil
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import save_image

from data.datasets import dataloader_from_config
from losses.combined_loss import CombinedLoss
from models.srrt import SRRT
from utils.ablation import ABLATIONS, apply_ablation
from utils.seed import collect_environment, set_seed

OOM_ADVICE = (
    "CUDA out of memory.\n"
    "  Activation memory scales with crop_size^2 and batch size.\n"
    "  Measured on an 8GB RTX 4070 Laptop (Restormer, AMP, + perceptual loss):\n"
    "    crop 128, batch 2 -> ~3.4 GB      crop 128, batch 4 -> ~6.4 GB\n"
    "    crop 256, batch 1 -> ~6.2 GB      crop 256, batch 2 -> OOM\n"
    "  Try, in configs/srrt_base.yaml:\n"
    "    - lower dataset.train.batch_size (raise training.grad_accum_steps to\n"
    "      keep the effective batch size unchanged)\n"
    "    - lower dataset.train.crop_size\n"
    "    - lower dataset.val.crop_size (validation runs at a larger crop)\n"
    "    - lower training.qualitative_samples\n"
    "    - ensure device.mixed_precision is true"
)


def build_optimizer(cfg: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    opt_cfg = cfg["optimizer"]
    if opt_cfg.get("type", "AdamW") != "AdamW":
        raise ValueError(f"Unsupported optimizer {opt_cfg['type']!r} (only AdamW is implemented)")

    return AdamW(
        model.parameters(),
        lr=opt_cfg["learning_rate"],
        weight_decay=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
    )


def build_scheduler(cfg: dict[str, Any], optimizer: torch.optim.Optimizer):
    sch_cfg = cfg["scheduler"]
    if sch_cfg.get("type", "CosineAnnealingLR") != "CosineAnnealingLR":
        raise ValueError(f"Unsupported scheduler {sch_cfg['type']!r}")

    return CosineAnnealingLR(
        optimizer,
        T_max=sch_cfg.get("T_max", cfg["training"]["epochs"]),
        eta_min=sch_cfg.get("eta_min", 0.0),
    )


def report_schedule(cfg, optimizer_lr: float, steps_per_epoch: int, grad_accum: int) -> dict:
    """
    Print the LR schedule BEFORE training so a misconfigured T_max is visible
    immediately rather than discovered after hours of compute.

    Builds a throwaway optimizer/scheduler identical to the real one and steps
    it forward to read off the actual LR at chosen epochs -- reporting the
    formula rather than the object would not catch a wiring mistake.
    """
    epochs = cfg["training"]["epochs"]
    t_max = cfg["scheduler"].get("T_max")
    total_opt_steps = (steps_per_epoch // grad_accum) * epochs

    probe = torch.nn.Parameter(torch.zeros(1))
    opt = AdamW([probe], lr=optimizer_lr,
                weight_decay=cfg["optimizer"]["weight_decay"],
                betas=tuple(cfg["optimizer"].get("betas", (0.9, 0.999))))
    sch = build_scheduler(cfg, opt)

    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(epochs):
        opt.step()
        sch.step()
        lrs.append(opt.param_groups[0]["lr"])

    probes = sorted({1, epochs // 2, epochs - 1, epochs})
    print("-" * 70)
    print("LR SCHEDULE VERIFICATION")
    print(f"  total epochs         : {epochs}")
    print(f"  steps per epoch      : {steps_per_epoch} iterations "
          f"({steps_per_epoch // grad_accum} optimizer steps, grad_accum={grad_accum})")
    print(f"  total optimizer steps: {total_opt_steps}")
    print(f"  scheduler            : {cfg['scheduler']['type']} "
          f"T_max={t_max} eta_min={cfg['scheduler'].get('eta_min', 0.0)}")
    print(f"  stepped              : once per EPOCH")
    print(f"  initial LR           : {lrs[0]:.6e}")
    for e in probes:
        if e <= epochs:
            print(f"  expected LR @ epoch {e:<3}: {lrs[e]:.6e}")

    ok = t_max == epochs
    print(f"  T_max == epochs      : {ok}"
          + ("" if ok else f"   <-- MISMATCH: LR would reach eta_min at epoch {t_max}, not {epochs}"))
    print("-" * 70)
    if not ok:
        raise SystemExit(
            f"Scheduler misconfigured: T_max={t_max} but training runs {epochs} epochs. "
            f"Pass --epochs (which retunes T_max) or fix scheduler.T_max in the config."
        )
    return {"epochs": epochs, "t_max": t_max, "steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": total_opt_steps, "lr_by_epoch": lrs}


def model_output(model, x: torch.Tensor) -> torch.Tensor:
    """
    Restored image from either model type.

    SRRT.forward returns a dict (it must also expose latents and loss terms);
    the bare Restormer backbone returns a Tensor. Accepting both lets validate()
    and the qualitative saver be reused with the plain baseline, which the audit
    and sanity scripts do.
    """
    out = model(x)
    return out["output"] if isinstance(out, dict) else out


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Mean PSNR over the batch. Inputs assumed in [0, 1]."""
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    return (10 * torch.log10((max_val ** 2) / mse.clamp(min=1e-10))).mean().item()


def checkpoint_state(model, optimizer, scheduler, scaler, epoch, best, cfg, env) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_val_psnr": best,
        "config": cfg,
        "environment": env,
    }


@torch.no_grad()
def validate(model, loader, loss_fn, device, max_batches=None,
             lpips_fn=None) -> dict[str, float]:
    """
    Validation metrics. SSIM and LPIPS are tracked per epoch (not just at the
    end) so convergence can be judged on perceptual quality, not PSNR alone.

    SSIM runs on CPU via scikit-image; LPIPS is optional so the cost can be
    dropped if it ever becomes significant.
    """
    from evaluate import compute_ssim

    model.eval()
    total_loss = total_psnr = total_ssim = total_lpips = 0.0
    n = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        inp = batch["input"].to(device, non_blocking=True)
        tgt = batch["target"].to(device, non_blocking=True)

        # No target passed: validation measures pure inference (the ODE path),
        # exactly as evaluation and deployment would.
        pred = model_output(model, inp)
        loss, _ = loss_fn(pred, tgt)

        clamped = pred.float().clamp(0, 1)
        bs = inp.size(0)
        total_loss += loss.item() * bs
        total_psnr += compute_psnr(clamped, tgt) * bs

        for b in range(bs):
            total_ssim += compute_ssim(clamped[b].cpu(), tgt[b].cpu())

        if lpips_fn is not None:
            total_lpips += float(lpips_fn(clamped * 2 - 1, tgt * 2 - 1).sum())

        n += bs

    model.train()
    d = max(n, 1)
    return {"loss": total_loss / d, "psnr": total_psnr / d, "ssim": total_ssim / d,
            "lpips": (total_lpips / d) if lpips_fn is not None else None}


@torch.no_grad()
def save_qualitative_samples(model, loader, device, output_dir, epoch, n_samples) -> None:
    """Save a degraded / restored / ground-truth comparison grid."""
    model.eval()

    inputs, targets = [], []
    for batch in loader:
        inputs.append(batch["input"])
        targets.append(batch["target"])
        if sum(t.shape[0] for t in inputs) >= n_samples:
            break

    inp = torch.cat(inputs)[:n_samples].to(device)
    tgt = torch.cat(targets)[:n_samples].to(device)

    # One image at a time: a batched forward at validation resolution is a
    # significant VRAM spike on a laptop GPU.
    preds = torch.cat([model_output(model, inp[i:i + 1]).clamp(0, 1).cpu()
                       for i in range(inp.shape[0])])

    grid = torch.cat([inp.cpu(), preds, tgt.cpu()], dim=0)
    save_image(grid, os.path.join(output_dir, f"epoch_{epoch:04d}_samples.png"),
               nrow=max(inp.shape[0], 1))

    model.train()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SRRT / the Restormer baseline.")
    parser.add_argument("--config", default="configs/srrt_base.yaml")
    parser.add_argument("--ablation", default=None, choices=sorted(ABLATIONS),
                        help="apply an ablation preset (overrides model.modules.*.enabled)")
    parser.add_argument("--resume", default=None, help="checkpoint path to resume from")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="stop after N optimizer steps (smoke testing)")
    parser.add_argument("--max-val-batches", type=int, default=None,
                        help="cap validation to N batches (overrides training.val_max_batches)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override training.epochs; also retunes scheduler T_max "
                             "so the cosine schedule completes over the run actually performed")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.ablation:
        cfg = apply_ablation(cfg, args.ablation)

    if args.epochs is not None:
        # Retune T_max too: a cosine schedule sized for 200 epochs would
        # barely decay over a short run, which is not the same experiment.
        cfg["training"]["epochs"] = args.epochs
        cfg["scheduler"]["T_max"] = args.epochs
        print(f"NOTE: epochs overridden to {args.epochs}; scheduler T_max retuned to match.")

    exp_cfg = cfg["experiment"]
    set_seed(exp_cfg["seed"], deterministic=exp_cfg.get("deterministic", False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = collect_environment()
    print(f"Experiment : {exp_cfg['name']}")
    print(f"Device     : {device} ({env.get('gpu_name', 'CPU')})")

    if device.type == "cuda":
        # Cap the allocator to real dedicated VRAM so we get a clean, catchable
        # OutOfMemoryError instead of silently spilling into Windows shared GPU
        # memory (system RAM), which is slow and has destabilized this machine.
        torch.cuda.set_per_process_memory_fraction(
            float(cfg["device"].get("max_memory_fraction", 0.85)), device.index or 0
        )

    # Per-experiment output directories so ablation runs never overwrite each other.
    run_name = exp_cfg["name"]
    save_dir = os.path.join(exp_cfg["save_dir"], run_name)
    log_dir = os.path.join(exp_cfg["log_dir"], run_name)
    output_dir = os.path.join(exp_cfg["output_dir"], run_name)
    for d in (save_dir, log_dir, output_dir):
        os.makedirs(d, exist_ok=True)

    train_loader = dataloader_from_config(cfg, "train")
    val_loader = dataloader_from_config(cfg, "val")
    print(f"Train/val  : {len(train_loader.dataset)} / {len(val_loader.dataset)} images")

    model = SRRT.from_config(cfg).to(device)
    loss_fn = CombinedLoss.from_config(cfg).to(device)

    summary = model.module_summary()
    active = [k for k in ("seecoder", "diffae", "lcr", "rectified_flow", "sra") if summary[k]["enabled"]]
    print(f"Modules    : {active if active else 'none (pure Restormer baseline)'}")
    print(f"Parameters : {summary['total']:,}")

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    use_amp = bool(cfg["device"].get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    train_cfg = cfg["training"]
    grad_accum_steps = max(int(train_cfg.get("grad_accum_steps", 1)), 1)
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))
    val_interval = int(train_cfg.get("val_interval", 1))
    save_interval = int(train_cfg.get("save_interval", 5))
    log_interval = int(train_cfg.get("log_interval", 20))
    n_qualitative = int(train_cfg.get("qualitative_samples", 4))

    # Per-epoch LPIPS. Frozen and eval-only; this is a METRIC here, never a
    # training objective (training uses VGG19 -- see docs/design_decisions.md D5).
    lpips_fn = None
    if "LPIPS" in cfg["evaluation"].get("metrics", []):
        import lpips as _lpips
        lpips_fn = _lpips.LPIPS(net=cfg["evaluation"].get("lpips_net", "alex"),
                                verbose=False).to(device).eval()
        for _p in lpips_fn.parameters():
            _p.requires_grad_(False)

    schedule_info = report_schedule(cfg, cfg["optimizer"]["learning_rate"],
                                    len(train_loader), grad_accum_steps)

    process = psutil.Process()
    peak_rss_mb = process.memory_info().rss / 1024 ** 2

    start_epoch, best_val_psnr = 0, -float("inf")

    if args.resume:
        # weights_only=False: our checkpoints intentionally embed the config dict
    # and environment metadata (spec section 24), not just tensors.
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_psnr = ckpt["best_val_psnr"]
        print(f"Resumed    : {args.resume} -> epoch {start_epoch} (best {best_val_psnr:.2f}dB)")

    writer = None
    if cfg.get("logging", {}).get("tensorboard", False):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)

    log_path = os.path.join(log_dir, "train_log.csv")
    write_header = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(["epoch", "train_loss", "train_l1", "train_perceptual",
                             "train_flow", "train_lcr", "val_loss", "val_psnr",
                             "val_ssim", "val_lpips", "lr",
                             "epoch_time_s", "iters_per_s", "peak_vram_mb", "peak_ram_mb"])
        log_file.flush()

    with open(os.path.join(log_dir, "run_metadata.json"), "w") as f:
        json.dump({"config": cfg, "environment": env, "module_summary": summary,
                   "schedule": schedule_info}, f, indent=2)

    global_step, stop_early = 0, False

    try:
        for epoch in range(start_epoch, train_cfg["epochs"]):
            epoch_start = time.time()
            model.train()
            running: dict[str, float] = {}
            n_seen = n_iters = 0

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            optimizer.zero_grad(set_to_none=True)

            for i, batch in enumerate(train_loader):
                inp = batch["input"].to(device, non_blocking=True)
                tgt = batch["target"].to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, enabled=use_amp):
                    # target passed so SRRT can compute the reference latent
                    # needed by the flow and LCR objectives.
                    out = model(inp, target=tgt)
                    loss, components = loss_fn(
                        out["output"], tgt,
                        flow_loss=out.get("flow_loss"),
                        lcr_loss=out.get("lcr_loss"),
                    )

                scaler.scale(loss / grad_accum_steps).backward()

                if (i + 1) % grad_accum_steps == 0:
                    if grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                bs = inp.size(0)
                n_seen += bs
                n_iters += 1
                for key, value in components.items():
                    running[key] = running.get(key, 0.0) + value.item() * bs

                if (i + 1) % log_interval == 0:
                    rate = n_iters / max(time.time() - epoch_start, 1e-9)
                    print(f"  epoch {epoch} iter {i + 1}/{len(train_loader)} "
                          f"loss={loss.item():.4f} ({rate:.2f} it/s)")
                    if writer is not None:
                        writer.add_scalar("train/loss_step", loss.item(), global_step)

                if args.max_steps is not None and global_step >= args.max_steps:
                    stop_early = True
                    break

            scheduler.step()
            averages = {k: v / max(n_seen, 1) for k, v in running.items()}
            train_loss = averages.get("total", 0.0)

            val_loss = val_psnr = val_ssim = val_lpips = None
            if (epoch + 1) % val_interval == 0 or stop_early:
                val_cap = (args.max_val_batches if args.max_val_batches is not None
                           else train_cfg.get("val_max_batches"))
                vm = validate(model, val_loader, loss_fn, device, val_cap, lpips_fn)
                val_loss, val_psnr = vm["loss"], vm["psnr"]
                val_ssim, val_lpips = vm["ssim"], vm["lpips"]
                print(f"[epoch {epoch}] val_loss={val_loss:.4f} val_psnr={val_psnr:.3f}dB "
                      f"val_ssim={val_ssim:.4f}"
                      + (f" val_lpips={val_lpips:.4f}" if val_lpips is not None else ""))

                if val_psnr > best_val_psnr:
                    best_val_psnr = val_psnr
                    torch.save(
                        checkpoint_state(model, optimizer, scheduler, scaler, epoch,
                                         best_val_psnr, cfg, env),
                        os.path.join(save_dir, "best_psnr.pth"),
                    )
                    print("           new best PSNR -> best_psnr.pth")

                if cfg.get("logging", {}).get("log_images", True):
                    save_qualitative_samples(model, val_loader, device, output_dir, epoch, n_qualitative)

            epoch_time = time.time() - epoch_start
            iters_per_s = n_iters / max(epoch_time, 1e-9)
            peak_vram = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                         if device.type == "cuda" else 0.0)
            lr = optimizer.param_groups[0]["lr"]

            peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / 1024 ** 2)
            log_writer.writerow([epoch, train_loss, averages.get("l1"), averages.get("perceptual"),
                                 averages.get("flow"), averages.get("lcr"),
                                 val_loss, val_psnr, val_ssim, val_lpips, lr,
                                 epoch_time, iters_per_s, peak_vram, peak_rss_mb])
            log_file.flush()

            if writer is not None:
                for key, value in averages.items():
                    writer.add_scalar(f"train/{key}", value, epoch)
                if val_psnr is not None:
                    writer.add_scalar("val/loss", val_loss, epoch)
                    writer.add_scalar("val/psnr", val_psnr, epoch)
                    writer.add_scalar("val/ssim", val_ssim, epoch)
                    if val_lpips is not None:
                        writer.add_scalar("val/lpips", val_lpips, epoch)
                writer.add_scalar("lr", lr, epoch)
                writer.add_scalar("system/peak_vram_mb", peak_vram, epoch)

            state = checkpoint_state(model, optimizer, scheduler, scaler, epoch,
                                     best_val_psnr, cfg, env)
            torch.save(state, os.path.join(save_dir, "latest.pth"))
            if (epoch + 1) % save_interval == 0:
                torch.save(state, os.path.join(save_dir, f"epoch_{epoch + 1:04d}.pth"))

            print(f"[epoch {epoch}] train_loss={train_loss:.4f} time={epoch_time:.1f}s "
                  f"({iters_per_s:.2f} it/s) peak_vram={peak_vram:.0f}MB")

            if stop_early:
                print(f"Stopping early: reached --max-steps={args.max_steps}")
                break

    except torch.cuda.OutOfMemoryError:
        print("\n" + "=" * 70 + f"\n{OOM_ADVICE}\n" + "=" * 70)
        raise
    finally:
        log_file.close()
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
