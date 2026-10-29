# Restormer Baseline — Results (VALIDATED, 60 epochs)

**Experiment:** `baseline_restormer`
**Status:** **VALIDATED — this is the baseline of record for all SRRT ablations**
**Run:** 60 epochs, seed 42, completed (epochs 0–59, cosine schedule run to completion)
**Best checkpoint:** **epoch 57** (`checkpoints/baseline_restormer/best_psnr.pth`)
**Sources:** `results/restormer_baseline/summary.json`,
`logs/baseline_restormer/train_log.csv`, `logs/baseline_restormer/run_metadata.json`.
Every number below is read from those measured artifacts.

> **Historical run preserved separately.** An earlier 10-epoch preliminary run
> (2026-08-17) is archived unmodified in
> [`BASELINE_RESULTS_10ep.md`](BASELINE_RESULTS_10ep.md). It was a pipeline-validation
> datapoint that stopped while validation PSNR was still rising, and it is **not**
> the ablation reference. Nothing in this document replaces or deletes it.

---

## 1. Headline metrics (authoritative)

GoPro **test** split, **1111 images**, `crop_mode=center`, `crop_size=256`,
evaluated from the epoch-57 checkpoint.

| Model | PSNR (dB) | SSIM | LPIPS |
|---|---|---|---|
| **Restormer (baseline, 60 epochs)** | **28.937 ± 3.857** | **0.8661 ± 0.0919** | **0.1356 ± 0.0841** |

Spread:

| Metric | mean | std | min | max | direction |
|---|---|---|---|---|---|
| PSNR | 28.937 | 3.857 | 17.823 | 40.529 | higher better |
| SSIM | 0.8661 | 0.0919 | 0.4237 | 0.9853 | higher better |
| LPIPS | 0.1356 | 0.0841 | 0.0092 | 0.7231 | LOWER better |

Hardest / easiest images (for consistent qualitative figure selection):

* worst 5 by PSNR: `71.png`, `24.png`, `150.png`, `192.png`, `275.png`
* best 5 by PSNR: `613.png`, `658.png`, `697.png`, `643.png`, `629.png`

**This is the reference row for every subsequent ablation.** Ablations must be
evaluated with the identical protocol (same split, same 1111 images, same
`center`/256 crop, same `evaluate.py`) or the comparison is invalid.

---

## 2. Training summary

| Quantity | Value |
|---|---|
| Epochs completed | 60 (0–59), full cosine schedule |
| Best val PSNR | **28.9136 dB @ epoch 57** |
| Final val PSNR | 28.9121 dB @ epoch 59 |
| Final val SSIM / LPIPS | 0.8529 / 0.1685 |
| Train loss, first → last | 0.22424 → 0.15509 |
| Val loss, first → last | 0.24375 → 0.17088 |
| Learning rate, first → last | 1.99864e-4 → 1.0e-6 (`eta_min` reached) |
| Optimizer steps | 15,720 (1051 iters/epoch ÷ 4 accum × 60) |
| Median epoch time | 352.3 s (5.87 min); min 298.0 s |
| Compute time | 6.08 h over 58 clean epochs |
| Median throughput | 2.983 it/s |
| Peak VRAM (training) | 5557 MB |
| Peak RAM (training) | 3438 MB |
| Inference time / image | 0.2183 s |
| Peak VRAM (evaluation) | 6784 MB |
| Parameters | **26,126,644** (backbone only; all SRRT modules disabled) |

**Two epoch times are not compute.** Epoch 9 (17,303 s) and epoch 12 (5,515 s)
record wall-clock across Windows Modern Standby, which suspended the machine
mid-run; the run was resumed from checkpoint. They are excluded from the 6.08 h
compute figure and from the median. `scripts/keep_awake.py` was added afterwards
to suppress idle standby for long runs. Pre-resume log copies are retained as
`train_log.csv.bak_pre_resume_*`.

### Convergence

The cosine schedule ran to completion: LR reached `eta_min = 1e-6` and validation
PSNR flattened over the final epochs (28.884 → 28.914 → 28.907 → 28.912 across
epochs 56–59, a spread of ~0.03 dB). The run therefore ended with the model at a
schedule-consistent optimum, unlike the 10-epoch run which was still climbing
steeply when its schedule expired.

---

## 3. Configuration and reproducibility

| Item | Value |
|---|---|
| Seed | 42 (`deterministic: false`) |
| Config | `configs/srrt_base.yaml`; full copy in `logs/baseline_restormer/run_metadata.json` |
| Ablation preset | `baseline_restormer` |
| **Modules** | seecoder ✗ · diffae ✗ · lcr ✗ · rectified_flow ✗ · sra ✗ — **all disabled** |
| Dataset | GoPro, `data/Gopro` (train 2103 pairs, test 1111 pairs) |
| Train crop | 128 px, `random`, augmentation on |
| Val/test crop | **256 px, `center`**, augmentation off |
| Batch / grad accum | 2 / 4 (effective batch 8) |
| Grad clip | 1.0 |
| Optimizer | AdamW, lr 2e-4, weight_decay 1e-4, betas (0.9, 0.999) |
| Scheduler | **CosineAnnealingLR, `T_max=60`, `eta_min=1e-6`** |
| AMP | enabled (`mixed_precision: true`), `max_memory_fraction: 0.85` |
| Loss weights | L1 1.0, perceptual 0.1 (VGG19); flow 0.1 and consistency 0.1 configured but **inactive** (modules disabled) |
| Checkpoints | `latest`, `best_psnr`, periodic every 10 epochs |
| Python / torch / CUDA | 3.11.9 / 2.11.0+cu128 / 12.8 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU (8187 MiB, cap 8.9) |
| Git hash | not a git repository |

`T_max` equals the epoch count, so the schedule is matched to the run rather
than truncated — the condition the 200-epoch YAML default would have violated.
`--epochs 60` retunes `T_max` to 60 at launch.

Reproduce with:

```bash
python train.py --ablation baseline_restormer --epochs 60
python evaluate.py --checkpoint checkpoints/baseline_restormer/best_psnr.pth \
                   --split test --out-dir results/restormer_baseline
```

---

## 4. Protocol caveats — read before citing these numbers

1. **Not comparable to published Restormer.** The paper reports ~32.9 dB on GoPro
   using **full images** and ~300k iterations with a progressive 128→384 crop
   schedule. This run is 15,720 optimizer steps at a fixed 128 px crop, evaluated
   on **256 px center crops**. The protocols differ in training length, crop
   schedule and evaluation geometry, so **this is not a Restormer reproduction and
   must not be presented as one.** For a protocol-comparable measurement use
   `evaluate.py --full-images`.
2. **Validation and test are the same split.** `dataset.val.split` is `test`, so
   there is no held-out third split. Best-checkpoint selection (epoch 57) used
   validation PSNR over `val_max_batches=150` batches of that split, and the
   headline metrics use all 1111 images of the same split. The reported figures
   are therefore *validation* numbers and may be mildly optimistic through model
   selection. The bias is bounded — selection was over 60 checkpoints on 150
   images — but it is real and is not corrected for.
3. **Validation during training ≠ final evaluation.** Per-epoch val PSNR
   (28.9136) is computed on 150 batches; the headline 28.937 dB uses all 1111
   images. The small difference is a sample-size effect, not a discrepancy.
4. **Single seed.** One run at seed 42. Run-to-run variance is unmeasured, so a
   small ablation delta cannot yet be called significant.
5. **GoPro / deblurring only.** Other datasets raise `NotImplementedError`.

---

## 5. Files

```text
results/restormer_baseline/
├── summary.json          aggregate metrics + module summary + environment
├── metrics.csv           per-image PSNR/SSIM/LPIPS for all 1111 test images
└── images/               8 deterministic qualitative samples
                          (ids 0, 1, 10, 100, 1000, 1001, 1002, 1003)

checkpoints/baseline_restormer/
├── best_psnr.pth         epoch 57  <- the baseline of record
├── latest.pth            epoch 59
└── epoch_00{10..60}.pth  periodic

logs/baseline_restormer/
├── train_log.csv         per-epoch metrics
├── run_metadata.json     full config + environment + LR schedule
└── events.out.tfevents.* TensorBoard
```

---

## 6. Honest assessment

* The 60-epoch run completed its full cosine schedule and validation PSNR
  flattened at the end. **This is a valid, converged-for-its-schedule baseline**
  and is the reference point for the SRRT ablation study.
* It is **not** a Restormer reproduction, and the gap to the published ~32.9 dB
  is explained by training length, crop schedule and evaluation geometry — not by
  a defect. Do not present it as matching the literature.
* Because validation and test are the same split and only one seed was run,
  **run-to-run variance is unknown**. Until it is measured, treat small ablation
  deltas as inconclusive rather than as improvements.
