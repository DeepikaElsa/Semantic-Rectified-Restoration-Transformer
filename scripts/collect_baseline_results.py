"""
Assemble the Phase 4/5 baseline deliverable.

Gathers ONLY numbers that were actually produced by a run:
  results/restormer_baseline/
      checkpoints/   -> pointer + metadata for best_psnr.pth and latest.pth
      metrics/       -> train_log.csv, summary.json, per-image metrics.csv
      images/        -> qualitative comparisons
      logs/          -> run_metadata.json (config + environment)
      training_curves.png

and writes research_notes/BASELINE_RESULTS.md from those files.

Nothing is estimated or interpolated: a metric that was not measured is written
as "not measured".

    python scripts/collect_baseline_results.py --experiment baseline_restormer
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="baseline_restormer")
    ap.add_argument("--out", default="results/restormer_baseline")
    ap.add_argument("--eval-limit", type=int, default=None,
                    help="evaluate only N test images (default: whole split)")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    exp = args.experiment
    ckpt_dir = os.path.join("checkpoints", exp)
    log_dir = os.path.join("logs", exp)
    out = args.out

    for sub in ("checkpoints", "metrics", "images", "logs"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    best = os.path.join(ckpt_dir, "best_psnr.pth")
    latest = os.path.join(ckpt_dir, "latest.pth")
    if not os.path.exists(best):
        print(f"No checkpoint at {best} -- has training run?")
        return 1

    # ---- training log ----
    train_log = os.path.join(log_dir, "train_log.csv")
    rows = []
    if os.path.exists(train_log):
        shutil.copy2(train_log, os.path.join(out, "metrics", "train_log.csv"))
        with open(train_log, newline="") as f:
            rows = list(csv.DictReader(f))

    meta_src = os.path.join(log_dir, "run_metadata.json")
    meta = {}
    if os.path.exists(meta_src):
        shutil.copy2(meta_src, os.path.join(out, "logs", "run_metadata.json"))
        with open(meta_src) as f:
            meta = json.load(f)

    # ---- final evaluation on the test split ----
    eval_dir = os.path.join(out, "metrics")
    if not args.skip_eval:
        cmd = [sys.executable, "evaluate.py", "--checkpoint", best,
               "--out-dir", eval_dir, "--save-images", "8"]
        if args.eval_limit:
            cmd += ["--limit", str(args.eval_limit)]
        print("$ " + " ".join(cmd))
        if subprocess.run(cmd).returncode != 0:
            print("evaluate.py failed")
            return 1

    summary = {}
    sp = os.path.join(eval_dir, "summary.json")
    if os.path.exists(sp):
        with open(sp) as f:
            summary = json.load(f)

    # move evaluation images into images/
    src_img = os.path.join(eval_dir, "images")
    if os.path.isdir(src_img):
        for fn in os.listdir(src_img):
            shutil.copy2(os.path.join(src_img, fn), os.path.join(out, "images", fn))
        shutil.rmtree(src_img, ignore_errors=True)

    # qualitative epoch grids
    qual = os.path.join("outputs", exp)
    if os.path.isdir(qual):
        for fn in sorted(os.listdir(qual)):
            shutil.copy2(os.path.join(qual, fn), os.path.join(out, "images", fn))

    # ---- checkpoint metadata (not the 300MB blobs) ----
    ck_meta = {}
    for name, path in (("best_psnr", best), ("latest", latest)):
        if os.path.exists(path):
            ck_meta[name] = {"path": path,
                             "size_mb": round(os.path.getsize(path) / 1024 ** 2, 1)}
    with open(os.path.join(out, "checkpoints", "checkpoints.json"), "w") as f:
        json.dump(ck_meta, f, indent=2)

    # ---- curves ----
    subprocess.run([sys.executable, "scripts/plot_training.py", "--experiment", exp,
                    "--out", os.path.join(out, "training_curves.png")])

    # ---- derive facts from the log ----
    n_epochs = len(rows)
    total_time_s = sum(to_float(r.get("epoch_time_s")) or 0.0 for r in rows)
    peak_vram = max([to_float(r.get("peak_vram_mb")) or 0.0 for r in rows], default=0.0)
    psnrs = [(int(r["epoch"]), to_float(r.get("val_psnr"))) for r in rows
             if to_float(r.get("val_psnr")) is not None]
    best_epoch, best_psnr = max(psnrs, key=lambda kv: kv[1]) if psnrs else (None, None)
    first_loss = to_float(rows[0]["train_loss"]) if rows else None
    last_loss = to_float(rows[-1]["train_loss"]) if rows else None

    def series(col):
        return [(int(r["epoch"]), to_float(r.get(col))) for r in rows
                if to_float(r.get(col)) is not None]

    ssim_pts, lpips_pts = series("val_ssim"), series("val_lpips")
    final_ssim = ssim_pts[-1][1] if ssim_pts else None
    best_ssim, best_ssim_ep = (max(ssim_pts, key=lambda kv: kv[1])[::-1]
                               if ssim_pts else (None, None))
    final_lpips = lpips_pts[-1][1] if lpips_pts else None
    best_lpips, best_lpips_ep = (min(lpips_pts, key=lambda kv: kv[1])[::-1]
                                 if lpips_pts else (None, None))
    final_psnr_val = psnrs[-1][1] if psnrs else None
    final_lr = to_float(rows[-1].get("lr")) if rows else None
    peak_ram = max([to_float(r.get("peak_ram_mb")) or 0.0 for r in rows], default=0.0)

    cfg = meta.get("config", {})
    env = meta.get("environment", {})

    def fmt(v, spec=".4f", missing="not measured"):
        return format(v, spec) if isinstance(v, (int, float)) else missing

    m = summary.get("metrics", {})
    psnr_s = m.get("psnr", {})
    ssim_s = m.get("ssim", {})
    lpips_s = m.get("lpips", {})

    # cfg is the RUN config, whose `epochs` was overwritten by --epochs. Read the
    # committed default from the YAML so the note does not claim the default is
    # whatever this short run happened to use.
    default_epochs = "n/a"
    try:
        import yaml as _yaml
        with open("configs/srrt_base.yaml") as _f:
            default_epochs = _yaml.safe_load(_f)["training"]["epochs"]
    except Exception:
        pass

    mean_epoch_min = (total_time_s / n_epochs / 60) if n_epochs else 0.0
    projected_h = (mean_epoch_min * default_epochs / 60
                   if isinstance(default_epochs, int) else None)

    converged_note = (
        f"**This run is NOT converged.** It ran **{n_epochs} epoch(s)**; the committed "
        f"config default is **{default_epochs}**"
        + (f", which at the measured {mean_epoch_min:.1f} min/epoch projects to "
           f"~{projected_h:.0f} h on this GPU" if projected_h else "")
        + ". Treat every number below as a pipeline-validation datapoint, not a "
          "research result and not a Restormer reproduction."
    )

    md = f"""# Restormer Baseline — Results

**Experiment:** `{exp}`
**Status:** baseline trained (bounded run), NOT converged
**Generated by:** `scripts/collect_baseline_results.py` from measured logs only

{converged_note}

---

## 1. Headline metrics (test split)

Evaluated on `{summary.get('split', 'n/a')}`, {summary.get('num_images', 'n/a')} images,
crop_mode=`{summary.get('crop_mode', 'n/a')}`, crop_size={summary.get('crop_size', 'n/a')}.

| Model | PSNR (dB) | SSIM | LPIPS |
|---|---|---|---|
| Restormer (baseline, {n_epochs} epochs) | {fmt(psnr_s.get('mean'), '.3f')} | {fmt(ssim_s.get('mean'))} | {fmt(lpips_s.get('mean'))} |

Spread (std / min / max):

| Metric | mean | std | min | max | direction |
|---|---|---|---|---|---|
| PSNR | {fmt(psnr_s.get('mean'), '.3f')} | {fmt(psnr_s.get('std'), '.3f')} | {fmt(psnr_s.get('min'), '.3f')} | {fmt(psnr_s.get('max'), '.3f')} | higher better |
| SSIM | {fmt(ssim_s.get('mean'))} | {fmt(ssim_s.get('std'))} | {fmt(ssim_s.get('min'))} | {fmt(ssim_s.get('max'))} | higher better |
| LPIPS | {fmt(lpips_s.get('mean'))} | {fmt(lpips_s.get('std'))} | {fmt(lpips_s.get('min'))} | {fmt(lpips_s.get('max'))} | LOWER better |

> Not comparable to published GoPro numbers: this evaluates
> `{summary.get('crop_mode', 'n/a')}` crops, and training used 128px patches
> (stage 1 of the paper's progressive schedule). Use `evaluate.py --full-images`
> for a protocol-comparable measurement.

## 2. Training summary

| Quantity | Value |
|---|---|
| Epochs completed | {n_epochs} |
| Final val PSNR | {fmt(final_psnr_val, '.4f')} dB (epoch {rows[-1]['epoch'] if rows else '?'}) |
| **Best val PSNR** | **{fmt(best_psnr, '.4f')} dB (epoch {best_epoch})** |
| Final val SSIM | {fmt(final_ssim)} |
| Best val SSIM | {fmt(best_ssim)} (epoch {best_ssim_ep if best_ssim_ep is not None else '-'}) |
| Final val LPIPS | {fmt(final_lpips)} |
| Best (lowest) val LPIPS | {fmt(best_lpips)} (epoch {best_lpips_ep if best_lpips_ep is not None else '-'}) |
| Final learning rate | {fmt(final_lr, '.3e')} |
| Peak system RAM | {peak_ram:.0f} MB |
| Train loss, first -> last epoch | {fmt(first_loss, '.5f')} -> {fmt(last_loss, '.5f')} |
| Total training time | {total_time_s / 60:.1f} min ({total_time_s / 3600:.2f} h) |
| Mean epoch time | {(total_time_s / n_epochs / 60) if n_epochs else 0:.2f} min |
| Peak VRAM (training) | {peak_vram:.0f} MB |
| Inference time / image | {fmt(summary.get('mean_inference_s'), '.4f')} s |
| Peak VRAM (evaluation) | {fmt(summary.get('peak_vram_mb'), '.0f')} MB |
| Parameters | {summary.get('parameters', 'n/a'):,} |

## 3. Reproducibility

| Item | Value |
|---|---|
| Seed | {cfg.get('experiment', {}).get('seed', 'n/a')} |
| Config | `configs/srrt_base.yaml` (full copy in `logs/run_metadata.json`) |
| Ablation preset | `{exp}` (all SRRT modules disabled) |
| Train crop / batch / accum | {cfg.get('dataset', {}).get('train', {}).get('crop_size', '?')}px / {cfg.get('dataset', {}).get('train', {}).get('batch_size', '?')} / {cfg.get('training', {}).get('grad_accum_steps', '?')} |
| Optimizer | {cfg.get('optimizer', {}).get('type', '?')} lr={cfg.get('optimizer', {}).get('learning_rate', '?')} wd={cfg.get('optimizer', {}).get('weight_decay', '?')} |
| Scheduler | {cfg.get('scheduler', {}).get('type', '?')} T_max={cfg.get('scheduler', {}).get('T_max', '?')} |
| AMP | {cfg.get('device', {}).get('mixed_precision', '?')} |
| Python / torch / CUDA | {env.get('python', '?')} / {env.get('torch', '?')} / {env.get('torch_cuda', '?')} |
| GPU | {env.get('gpu_name', '?')} ({env.get('gpu_total_memory_mb', '?')} MiB) |
| Git hash | {env.get('git_hash') or 'not a git repository'} |

Reproduce with:

```bash
python train.py --ablation {exp} --epochs {n_epochs}
python evaluate.py --checkpoint checkpoints/{exp}/best_psnr.pth
```

## 4. Files

```
{out}/
├── checkpoints/checkpoints.json     paths + sizes (blobs stay in checkpoints/{exp}/)
├── metrics/train_log.csv            per-epoch training metrics
├── metrics/metrics.csv              per-image test metrics
├── metrics/summary.json             aggregate metrics + environment
├── images/                          qualitative comparisons
├── logs/run_metadata.json           full config + environment snapshot
└── training_curves.png
```

## 5. Honest assessment

- The pipeline trains, checkpoints, resumes, validates and evaluates end to end.
- The numbers above are **real measurements from this run**, not estimates.
- They do **not** establish a converged Restormer baseline, and must not be used
  as the comparison point for SRRT ablations until a longer run is performed.
- No SRRT ablation has been run. No claim about LCR or any other module exists.
"""

    os.makedirs("research_notes", exist_ok=True)
    with open("research_notes/BASELINE_RESULTS.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\nWrote research_notes/BASELINE_RESULTS.md and {out}/")
    if psnrs:
        print(f"best val PSNR {best_psnr:.3f} dB at epoch {best_epoch} over {n_epochs} epochs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
