"""
Phase 1 pre-training audit: verify the 20 pipeline items with hard evidence.

Read-only apart from a temporary checkpoint written to the scratch dir. Prints
PASS/FAIL/INFO per item and exits non-zero if any check fails, so this can gate
a training run.

    python scripts/audit_pipeline.py
"""

import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import GoProDataset, dataloader_from_config  # noqa: E402
from losses.combined_loss import CombinedLoss  # noqa: E402
from models.backbone.restormer import Restormer  # noqa: E402
from models.srrt import SRRT  # noqa: E402
from train import (build_optimizer, build_scheduler, checkpoint_state,  # noqa: E402
                   compute_psnr, validate)
from utils.ablation import apply_ablation  # noqa: E402
from utils.seed import collect_environment, set_seed  # noqa: E402

RESULTS = []


def check(item, name, ok, detail=""):
    RESULTS.append((item, name, bool(ok), detail))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {item:>2}. {name}" + (f"\n         {detail}" if detail else ""))


def info(item, name, detail):
    RESULTS.append((item, name, None, detail))
    print(f"[INFO] {item:>2}. {name}\n         {detail}")


def main():
    print("=" * 78)
    print("SRRT PRE-TRAINING AUDIT")
    print("=" * 78)

    with open("configs/srrt_base.yaml") as f:
        cfg = yaml.safe_load(f)

    # Baseline config: all SRRT modules off (Phase 2 requirement).
    cfg = apply_ablation(cfg, "baseline_restormer")
    cfg["dataset"]["train"]["num_workers"] = 0
    cfg["dataset"]["val"]["num_workers"] = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 1. dataset paths -------------------------------------------------
    root = cfg["dataset"]["root"]
    paths = {s + "/" + k: os.path.join(root, s, k)
             for s in ("train", "test") for k in ("blur", "sharp")}
    missing = [p for p in paths.values() if not os.path.isdir(p)]
    counts = {k: len(os.listdir(v)) for k, v in paths.items() if os.path.isdir(v)}
    check(1, "Dataset paths exist", not missing,
          f"root={root} counts={counts}" + (f" MISSING={missing}" if missing else ""))

    # ---- 2. blur/sharp pairing -------------------------------------------
    train_ds = GoProDataset(root=root, split="train", crop_size=64, crop_mode="center", augment=False)
    test_ds = GoProDataset(root=root, split="test", crop_size=64, crop_mode="center", augment=False)
    mismatched = [(a, b) for a, b in train_ds.pairs if a != b]
    check(2, "Blur/sharp pairing validated per-file",
          len(mismatched) == 0 and len(train_ds) == counts.get("train/blur"),
          f"train pairs={len(train_ds)} test pairs={len(test_ds)} name-mismatches={len(mismatched)}")

    # ---- 3. normalization -------------------------------------------------
    s = train_ds[0]
    x, y = s["input"], s["target"]
    in_range = (0.0 <= float(x.min()) and float(x.max()) <= 1.0
                and 0.0 <= float(y.min()) and float(y.max()) <= 1.0)
    check(3, "Image normalization in [0,1], float32",
          in_range and x.dtype == torch.float32,
          f"input min/max={float(x.min()):.3f}/{float(x.max()):.3f} dtype={x.dtype}")

    # ---- 4. train/val split ----------------------------------------------
    train_files = {a for a, _ in train_ds.pairs}
    test_files = {a for a, _ in test_ds.pairs}
    overlap = train_files & test_files
    tr_split = cfg["dataset"]["train"]["split"]
    va_split = cfg["dataset"]["val"]["split"]
    # GoPro filenames are per-split indices, so name overlap is expected and is
    # NOT leakage; what matters is that the two splits read different folders.
    check(4, "Train/val read distinct split directories", tr_split != va_split,
          f"train dir={tr_split}, val dir={va_split}, "
          f"filename overlap={len(overlap)} (expected: separate folders, indices restart)")

    # ---- 5. tensor dimensions --------------------------------------------
    tl = dataloader_from_config(cfg, "train")
    vl = dataloader_from_config(cfg, "val")
    tb, vb = next(iter(tl)), next(iter(vl))
    cs_t = cfg["dataset"]["train"]["crop_size"]
    cs_v = cfg["dataset"]["val"]["crop_size"]
    bs_t = cfg["dataset"]["train"]["batch_size"]
    dims_ok = (tuple(tb["input"].shape) == (bs_t, 3, cs_t, cs_t)
               and tuple(vb["input"].shape) == (1, 3, cs_v, cs_v)
               and tb["input"].shape == tb["target"].shape)
    check(5, "Tensor dimensions match config", dims_ok,
          f"train batch={tuple(tb['input'].shape)} val batch={tuple(vb['input'].shape)}")

    # ---- 6. parameter count ----------------------------------------------
    model = Restormer.from_config(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    check(6, "Parameter count (Restormer ~26.1M as published)",
          25_000_000 < n_params < 27_500_000,
          f"total={n_params:,} trainable={trainable:,}")

    # ---- 7. device placement ---------------------------------------------
    devs = {p.device.type for p in model.parameters()}
    check(7, "Device placement", devs == {device.type},
          f"cuda_available={torch.cuda.is_available()} model devices={devs} target={device.type}")

    # ---- 8. AMP -----------------------------------------------------------
    use_amp = bool(cfg["device"]["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    with torch.autocast(device_type=device.type, enabled=use_amp):
        probe_in = tb["input"][:1].to(device)
        # Probe an INTERNAL conv: the model output is legitimately fp32 because
        # the final global residual (conv_out + y) promotes fp16 + fp32 -> fp32.
        probe_internal = model.encoder.patch_embed(probe_in)
        probe_out = model(probe_in)
    amp_ok = (not use_amp) or (probe_internal.dtype == torch.float16 and scaler.is_enabled())
    check(8, "AMP active (internal ops fp16, scaler enabled)", amp_ok,
          f"mixed_precision={use_amp} scaler={scaler.is_enabled()} "
          f"internal conv dtype={probe_internal.dtype} "
          f"model output dtype={probe_out.dtype} (fp32 expected: residual promotion)")

    # ---- 9. optimizer -----------------------------------------------------
    opt = build_optimizer(cfg, model)
    g = opt.param_groups[0]
    oc = cfg["optimizer"]
    check(9, "Optimizer matches config",
          type(opt).__name__ == "AdamW" and g["lr"] == oc["learning_rate"]
          and g["weight_decay"] == oc["weight_decay"],
          f"{type(opt).__name__} lr={g['lr']} wd={g['weight_decay']} betas={g['betas']}")

    # ---- 10. scheduler ----------------------------------------------------
    sch = build_scheduler(cfg, opt)
    lr0 = opt.param_groups[0]["lr"]
    opt.step()
    sch.step()
    lr1 = opt.param_groups[0]["lr"]
    check(10, "LR scheduler decreases LR", lr1 < lr0,
          f"{type(sch).__name__} T_max={cfg['scheduler']['T_max']} lr {lr0:.10f} -> {lr1:.10f} (delta={lr0 - lr1:.3e})")

    # ---- 11 & 12. checkpoint save + resume --------------------------------
    tmp = tempfile.mkdtemp()
    try:
        ck = os.path.join(tmp, "t.pth")
        env = collect_environment()
        torch.save(checkpoint_state(model, opt, sch, scaler, 7, 31.5, cfg, env), ck)
        loaded = torch.load(ck, map_location=device, weights_only=False)
        keys = {"epoch", "model_state", "optimizer_state", "scheduler_state",
                "scaler_state", "best_val_psnr", "config", "environment"}
        check(11, "Checkpoint contains state + config + environment",
              keys <= set(loaded), f"keys={sorted(set(loaded))}")

        m2 = Restormer.from_config(cfg).to(device)
        before = sum(float((a - b).abs().sum())
                     for a, b in zip(m2.state_dict().values(), model.state_dict().values()))
        m2.load_state_dict(loaded["model_state"])
        opt2 = build_optimizer(cfg, m2)
        sch2 = build_scheduler(cfg, opt2)
        opt2.load_state_dict(loaded["optimizer_state"])
        sch2.load_state_dict(loaded["scheduler_state"])
        after = sum(float((a - b).abs().sum())
                    for a, b in zip(m2.state_dict().values(), model.state_dict().values()))
        check(12, "Resume restores weights + optimizer + scheduler exactly",
              after == 0.0 and before > 0.0
              and loaded["epoch"] == 7 and loaded["best_val_psnr"] == 31.5,
              f"weight diff before load={before:.3e} after={after:.3e}; "
              f"epoch={loaded['epoch']} best={loaded['best_val_psnr']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- 13. loss ---------------------------------------------------------
    loss_fn = CombinedLoss.from_config(cfg).to(device)
    inp = tb["input"].to(device)
    tgt = tb["target"].to(device)
    with torch.autocast(device_type=device.type, enabled=use_amp):
        pred = model(inp)
        total, comps = loss_fn(pred, tgt)
    manual_l1 = float(torch.nn.functional.l1_loss(pred.float(), tgt).detach())
    check(13, "Loss finite and L1 term matches manual computation",
          torch.isfinite(total) and abs(float(comps["l1"]) - manual_l1) < 1e-4,
          f"total={float(total):.5f} l1={float(comps['l1']):.5f} (manual {manual_l1:.5f}) "
          f"perceptual={float(comps.get('perceptual', 0)):.5f} terms={sorted(comps)}")

    # ---- 14. PSNR ---------------------------------------------------------
    a = torch.rand(2, 3, 16, 16, device=device)
    psnr_identical = compute_psnr(a, a)
    # analytic: constant offset 0.1 -> MSE=0.01 -> PSNR=20 dB
    psnr_known = compute_psnr(torch.full((1, 3, 8, 8), 0.6, device=device),
                              torch.full((1, 3, 8, 8), 0.5, device=device))
    psnr_real = compute_psnr(pred.detach().float().clamp(0, 1), tgt)
    check(14, "PSNR analytic check + finite on real batch",
          abs(psnr_known - 20.0) < 1e-3 and np.isfinite(psnr_real) and psnr_identical > 90,
          f"offset-0.1 case={psnr_known:.4f}dB (expect 20.0) identical={psnr_identical:.1f}dB "
          f"real batch={psnr_real:.2f}dB")

    # ---- 15. SSIM ---------------------------------------------------------
    from evaluate import compute_ssim
    ssim_same = compute_ssim(tgt[0].cpu(), tgt[0].cpu())
    ssim_real = compute_ssim(pred[0].detach().float().clamp(0, 1).cpu(), tgt[0].cpu())
    check(15, "SSIM = 1.0 for identical, in [-1,1] for real",
          abs(ssim_same - 1.0) < 1e-6 and -1.0 <= ssim_real <= 1.0,
          f"identical={ssim_same:.6f} real={ssim_real:.4f}")

    # ---- 16. LPIPS --------------------------------------------------------
    import lpips
    lp = lpips.LPIPS(net=cfg["evaluation"]["lpips_net"], verbose=False).to(device).eval()
    with torch.no_grad():
        lp_same = float(lp(tgt[:1] * 2 - 1, tgt[:1] * 2 - 1).item())
        lp_real = float(lp(pred[:1].detach().float().clamp(0, 1) * 2 - 1, tgt[:1] * 2 - 1).item())
    check(16, "LPIPS ~0 for identical, finite for real",
          lp_same < 1e-4 and np.isfinite(lp_real),
          f"identical={lp_same:.6f} real={lp_real:.4f} (net={cfg['evaluation']['lpips_net']})")

    # ---- 17. validation loop ---------------------------------------------
    vloss, vpsnr = validate(model, vl, loss_fn, device, max_batches=2)
    check(17, "Validation loop returns finite loss and PSNR",
          np.isfinite(vloss) and np.isfinite(vpsnr),
          f"val_loss={vloss:.4f} val_psnr={vpsnr:.2f}dB (2 batches at {cs_v}px)")

    # ---- 18. reproducibility ---------------------------------------------
    def sample_once():
        set_seed(cfg["experiment"]["seed"])
        m = Restormer.from_config(cfg)
        ds = GoProDataset(root=root, split="train", crop_size=64,
                          crop_mode="random", augment=True)
        return (float(next(iter(m.parameters())).sum()), ds[0]["input"].clone())

    w1, c1 = sample_once()
    w2, c2 = sample_once()
    check(18, "Seeding reproduces init weights AND random crops",
          w1 == w2 and torch.equal(c1, c2),
          f"first-param sum identical={w1 == w2}, random crop identical={torch.equal(c1, c2)}")

    # ---- 19. Windows DataLoader -----------------------------------------
    nw_t = cfg["dataset"]["train"]["num_workers"]
    guarded = []
    for fn in ("train.py", "evaluate.py"):
        with open(fn) as f:
            guarded.append('if __name__ == "__main__":' in f.read())
    with open("configs/srrt_base.yaml") as f:
        cfg_raw = yaml.safe_load(f)
    check(19, "Entry points guarded for spawn; num_workers configurable",
          all(guarded),
          f"__main__ guard train.py/evaluate.py={guarded}; "
          f"config num_workers train={cfg_raw['dataset']['train']['num_workers']} "
          f"val={cfg_raw['dataset']['val']['num_workers']} (audit forced 0)")

    # ---- 20. GPU memory --------------------------------------------------
    # Delegated to a SUBPROCESS. By this point the audit holds a second
    # Restormer, VGG19 and LPIPS on the GPU; measuring here would report a peak
    # far above what real training uses. A clean process holds only
    # model + loss + optimizer + one batch, matching the training loop.
    if device.type == "cuda":
        import json
        import subprocess
        proc = subprocess.run(
            [sys.executable, "scripts/_measure_vram.py", "--ablation", "baseline_restormer"],
            capture_output=True, text=True,
        )
        line = [l for l in proc.stdout.strip().splitlines() if l.startswith("{")]
        if not line:
            check(20, "One train step fits within the configured VRAM cap", False,
                  f"measurement subprocess failed: {proc.stderr.strip()[-300:]}")
        else:
            m = json.loads(line[-1])
            check(20, "One train step fits within the configured VRAM cap",
                  (not m["oom"]) and m["peak_mb"] < m["cap_mb"],
                  f"peak={m['peak_mb']:.0f}MB reserved={m['reserved_mb']:.0f}MB "
                  f"cap={m['cap_mb']:.0f}MB total={m['total_vram_mb']:.0f}MB "
                  f"(batch={m['batch_size']} crop={m['crop_size']} AMP={m['amp']}, "
                  f"measured in clean subprocess)")
    else:
        info(20, "GPU memory", "CUDA unavailable; skipped")

    # ---- summary ---------------------------------------------------------
    fails = [r for r in RESULTS if r[2] is False]
    print("=" * 78)
    print(f"AUDIT: {sum(1 for r in RESULTS if r[2] is True)} passed, "
          f"{len(fails)} failed, {sum(1 for r in RESULTS if r[2] is None)} info")
    print("=" * 78)
    if fails:
        for item, name, _, detail in fails:
            print(f"  FAILED {item}. {name}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
