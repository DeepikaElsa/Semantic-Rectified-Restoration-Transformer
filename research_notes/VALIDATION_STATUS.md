# SRRT — Validation Status

**Date:** 2026-08-28 (baseline validated; superseding the 2026-08-14 revision)
**Machine:** Windows 11, RTX 4070 Laptop (8187 MiB), Python 3.11.9, torch 2.11.0+cu128 (CUDA runtime 12.8, driver CUDA 12.9)

This file separates three things that are easy to conflate:

* **VERIFIED** — checked by an executed test/measurement in this repository
* **IMPLEMENTED ONLY** — code exists and runs, but nothing confirms it is *correct
  or useful* beyond shapes and finiteness
* **NOT VALIDATED** — no experimental evidence exists

> **A validated Restormer baseline now exists (60 epochs, PSNR 28.937 dB — see
> §5).** That is a *baseline*, not an SRRT result. **SRRT has still not been shown
> to work, and no claim that it improves on Restormer is supported by anything in
> this repository.**

---

## 1. VERIFIED (executed evidence)

### 1.1 Pre-training audit — 20/20 pass

`python scripts/audit_pipeline.py` (exits non-zero on failure, so it can gate a run).

| # | Item | Evidence |
|---|---|---|
| 1 | Dataset paths | `data/Gopro`: train blur/sharp 2103/2103, test 1111/1111 |
| 2 | Blur/sharp pairing | 2103 train + 1111 test pairs, 0 filename mismatches, validated per file |
| 3 | Normalization | float32 in [0,1] (sample min/max 0.110/0.686) |
| 4 | Train/val split | train reads `train/`, val reads `test/` — distinct directories |
| 5 | Tensor dimensions | train `(2,3,128,128)`, val `(1,3,256,256)`, input.shape == target.shape |
| 6 | Parameter count | 26,126,644 total / trainable (matches published Restormer ~26.1 M) |
| 7 | Device placement | all params on `cuda`; `torch.cuda.is_available()` True |
| 8 | AMP | internal conv dtype `float16`, GradScaler enabled. Model *output* is fp32 — correct: the final global residual `conv + y` promotes fp16+fp32→fp32 |
| 9 | Optimizer | AdamW lr=2e-4 wd=1e-4 betas=(0.9,0.999), matches YAML |
| 10 | Scheduler | CosineAnnealingLR T_max=200: 0.0002000000 → 0.0001999877 after 1 step |
| 11 | Checkpoint contents | epoch, model/optimizer/scheduler/scaler state, best metric, **config**, **environment** |
| 12 | Resume | weight diff before load 9.96e+05 → **0.000e+00** after; epoch/best restored |
| 13 | Loss | finite; L1 term matches an independent manual computation to 1e-4 |
| 14 | PSNR | analytic case (constant 0.1 offset) = **20.0000 dB** exactly as theory; identical images 100 dB |
| 15 | SSIM | identical images = **1.000000**; real pair in range |
| 16 | LPIPS | identical images = **0.000000**; real pair finite (net=alex) |
| 17 | Validation loop | returns finite loss and PSNR |
| 18 | Reproducibility | same seed reproduces **both** init weights and random crops |
| 19 | Windows DataLoader | `if __name__ == "__main__":` present in train.py and evaluate.py; num_workers YAML-configurable |
| 20 | GPU memory | one train step peak **2201 MB** vs 6959 MB cap (measured in a clean subprocess) |

### 1.2 Zero-gate equivalence (Phase 7) — holds EXACTLY

`python scripts/zero_gate_check.py` → `results/zero_gate/zero_gate_128.json`

Output of every ablation configuration vs. the bare Restormer, identical backbone weights, 1×3×128×128 input, float32:

| Configuration | max abs error | mean abs error | relative error |
|---|---|---|---|
| baseline_restormer | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| seecoder | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| seecoder_diffae | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| seecoder_diffae_lcr | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| seecoder_diffae_lcr_flow | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| full_srrt | 0.000e+00 | 0.000e+00 | 0.000e+00 |

Stronger than the expected "≈": the gates are *exactly* zero at init
(`gamma_latent=0`, `sra.gamma=0`, `flow.proj_out.weight/bias=0`) and the flow
velocity field is identically zero, so the ODE is the identity map and the
injected terms contribute exactly nothing. Outputs are bitwise identical.

**Why this matters:** every ablation stage starts from the *same* function as
the baseline. Any difference measured later is attributable to what a module
learned, not to initialization noise it introduced.

**Re-verified 2026-08-28** after the SeeCoder injection fix (§4): the table above
is unchanged, all six configurations still bitwise-identical at init.

**Limit of this check.** Identical output at init does *not* prove a module is
connected. A module behind a zero gate and a module wired to nothing both score
0.000e+00 here. Connectivity is now covered separately by
`tests/test_srrt.py::test_enabled_modules_are_actually_connected`, which asserts
that every module an ablation enables receives non-zero gradient from the
training objective after the gate leaves zero.

### 1.3 Sanity test (Phase 2) — 8/8 pass

`python scripts/sanity_check.py --iters 10 --patch 64` → `results/sanity_check/sanity.json`
Pure Restormer, batch 1, 64px, AMP, one fixed batch:

| Iteration | Loss | PSNR | GPU VRAM |
|---|---|---|---|
| 1 | 0.256482 | 22.725 dB | 1063 MB |
| 4 | 0.245911 | 24.640 dB | 1065 MB |
| 6 | 0.189529 | 26.607 dB | 1065 MB |
| 8 | 0.198053 | 28.714 dB | 1065 MB |
| 10 | 0.140192 | 30.366 dB | 1065 MB |

Forward finite · loss finite · gradients finite · **494/494 params updated** ·
loss decreased · PSNR finite · VRAM 1065 MB < 6959 MB cap · CPU RSS stable.

### 1.4 Overfit test (Phase 3) — 6/6 pass

`python scripts/overfit_test.py --num-images 16 --epochs 40 --patch 128`
→ `results/sanity_overfit/` (metrics.csv, summary.json, curves.png, images/)

16 fixed pairs, batch 1, center crop, **no augmentation**, seed 42:

| | L1 | PSNR | SSIM |
|---|---|---|---|
| before training | 0.078904 | 20.298 dB | 0.4363 |
| after 40 epochs | 0.025427 | **29.579 dB** | 0.8581 |
| change | −0.053 | **+9.281 dB** | +0.422 |

640 iterations, 410 s, peak VRAM 2006 MB. Qualitative strips
(`images/sample*_comparison.png`, order: input | initial pred | final pred |
target) show the prediction moving from untrained noise to a visibly sharper,
colour-correct image.

**This is a debugging result, not a benchmark.** It establishes only that the
data→model→loss→optimizer chain can learn.

### 1.5 Measured performance

`python scripts/measure_throughput.py` → `results/throughput.json`

| Quantity | Measured |
|---|---|
| Train steady state (batch 2, 128px, AMP, 4 workers) | **2.7–3.0 it/s (333–372 ms/iter)** |
| Iterations per epoch | 1051 |
| Train-only epoch | ~6.5 min |
| Validation | 198 ms/image at 256px → 3.7 min for all 1111 |
| Full epoch (train + full val) | ~10.2 min |
| **200 epochs (config default)** | **~34 hours** |
| Peak VRAM, one training step | 2201 MB (cap 6959 MB) |

> **Correction to an earlier note in `research_notes/HISTORY.md`:** a previous session
> reported "0.05–0.2 it/s". That figure was measured over a 3-step run and was
> dominated by process startup, validation and 300 MB checkpoint writes.
> Steady-state throughput is **2.7–3.0 it/s**, ~15× faster. Training on this
> laptop is far more practical than previously stated.

### 1.6 Unit tests — 171 pass

`python -m pytest tests/ -q`. Includes the MDTA HW×HW regression guard (intercepts
`torch.matmul` and fails on any oversized square intermediate), applied across
attention/encoder/decoder/Restormer/LCR/SRA/SRRT.

---

## 2. IMPLEMENTED ONLY (runs; correctness/benefit unproven)

* **SeeCoder** (335,380 params) — produces multi-scale features + context at H/8. Shapes and gradient flow tested. No evidence it captures useful "visual context".
* **DiffAE** (2,322,080) — emits spatial + semantic latents. **No disentanglement guarantee**; it is a *learned damage-aware representation* only.
* **LCR** (5,128,236) — fusion + consistency loss verified mathematically (loss is exactly 0 for identical latents; L2-normalized projections make it scale-invariant, so it cannot be gamed by shrinking norms; z_ref is detached). **Whether it improves restoration is entirely unknown — this is the central open question.**
* **Rectified Flow** (4,067,968) — interpolation endpoints exact, `L_flow` equals ‖z1−z0‖² at init (v≡0), Euler integration exact for constant velocity. Never trained, so `v_θ` has learned nothing.
* **SRA** (68,453) — exact no-op at init; gate in (0,1). Untrained.
* **Full SRRT** (37,235,130) — all 6 ablations run forward+backward.
* **evaluate.py** — PSNR/SSIM/LPIPS + CSV/JSON/images; exercised only on an untrained checkpoint.
* **Ablation framework, TensorBoard/CSV logging, checkpoint+resume, OOM handling.**

---

## 3. NOT VALIDATED (no experimental evidence)

1. **No SRRT result of any kind.** A validated *Restormer* baseline now exists
   (§5), but no SRRT module has produced a measured outcome.
2. **Run-to-run variance is unmeasured.** The baseline is a single seed (42).
   Until variance is quantified, a small ablation delta cannot be called an
   improvement — see `BASELINE_RESULTS.md` §4.
3. **Ablation 1 (`seecoder`) is IN PROGRESS, not complete.** No numbers exist for
   it and none may be cited. Experiments 3–6 have not been started, by design.
4. **No evidence LCR contributes anything.** It is the proposed novelty and is completely unmeasured.
5. **Loss weights untuned** — `(1.0, 0.1, 0.1, 0.1)` are spec defaults.
6. **Evaluation protocol is not literature-comparable.** Validation uses 256px center crops; published GoPro numbers use full images (`evaluate.py --full-images`). Training also uses 128px crops, the *first* stage of Restormer's progressive schedule, so results should not be compared to the paper. The validated 60-epoch baseline (28.937 dB) is **not** a Restormer reproduction and must not be presented as matching the published ~32.9 dB.
7. **Train/inference mismatch in the flow path** (`docs/design_decisions.md` D9): training injects the supervised endpoint z1, inference injects the ODE solution. Exposure bias unaddressed.
8. **Only GoPro / deblurring.** The other five datasets raise `NotImplementedError`.
9. **Validation and test are the same split.** `dataset.val.split = test`, so best-checkpoint selection and the reported metrics use the same 1111 images. There is no held-out third split; reported figures may be mildly optimistic through model selection.

---

## 4. Bugs found and fixed during this validation pass

| Bug | Impact | Status |
|---|---|---|
| `compute_ssim` called `.numpy()` without `.detach()` | crashed on any grad-tracking tensor; latent because `evaluate.py` only calls it under `no_grad` | fixed |
| `validate()` / qualitative saver assumed a dict output | could not be reused with the bare `Restormer`; fixed with `model_output()` accepting either | fixed |
| Throwaway throughput probe lacked `if __name__ == "__main__":` | on Windows spawn each DataLoader worker re-ran the script and spawned its own workers → hang at 0% GPU. Confirms why spec §14 requires the guard; `train.py`/`evaluate.py` already had it | fixed (script moved to `scripts/measure_throughput.py`, guarded) |
| Audit VRAM measured with VGG19 + LPIPS + a second Restormer resident | reported 8683 MB, wrongly implying the config does not fit | fixed — measured in a clean subprocess (2201 MB) |
| Doc example asserted fp32 MSE to 1e-9 | test failed on exact-but-imprecise float | fixed tolerance |
| **SeeCoder had no path to the output** — `inject`/`gamma_latent` were built only when LCR was enabled, so `--ablation seecoder` (and `seecoder_diffae`) ran the module and discarded it. Measured: perturbing every SeeCoder weight by N(0,10) changed the output by **0.0**, and 0/44 SeeCoder tensors received gradient. The zero-gate check could not see it, because "gated" and "disconnected" both score 0.000e+00 | Ablations B and C would have reproduced the baseline exactly while reporting a 335,380-parameter module as "enabled" | fixed 2026-08-28 — injection now built whenever SeeCoder or DiffAE is on; zero-gate property re-verified as still exact; pinned by `test_enabled_modules_are_actually_connected` |

---

## 5. Current position

```text
IMPLEMENTED -> SANITY TESTED -> OVERFIT TESTED -> BASELINE TRAINED -> BASELINE VALIDATED -> SRRT ABLATIONS
    done           done             done           done (60 ep)          done               1 of 5 running
```

### 5.1 Baseline of record (Phase 4) — VALIDATED

**60 epochs**, `baseline_restormer`, seed 42, all SRRT modules disabled,
26,126,644 parameters. Best checkpoint **epoch 57**
(`checkpoints/baseline_restormer/best_psnr.pth`). Cosine schedule `T_max=60`
run to completion (LR reached `eta_min=1e-6`); validation PSNR flat over the
final four epochs (spread ~0.03 dB). 6.08 h compute.

| GoPro test split (1111 images, 256px center crop) | PSNR | SSIM | LPIPS |
|---|---|---|---|
| **Restormer, 60 epochs (epoch 57)** | **28.937 +/- 3.857 dB** | **0.8661 +/- 0.0919** | **0.1356 +/- 0.0841** |

**This is the reference row for every SRRT ablation.** Ablations must use the
identical protocol — same split, same 1111 images, same `center`/256 crop, same
`evaluate.py` — or the comparison is invalid.

Full configuration, reproducibility details and protocol caveats:
[`BASELINE_RESULTS.md`](BASELINE_RESULTS.md). Artifacts in
`results/restormer_baseline/`.

**Not a Restormer reproduction.** Published GoPro is ~32.9 dB on *full images*
after ~300k iterations with a progressive 128→384 crop schedule; this is 15,720
optimizer steps at a fixed 128px crop, scored on 256px center crops. The gap is
explained by protocol, not by a defect, and the two numbers must not be compared.

### 5.2 Historical 10-epoch preliminary run — superseded, retained

10 epochs, seed 42, 73.5 min. Archived unmodified in
[`BASELINE_RESULTS_10ep.md`](BASELINE_RESULTS_10ep.md); checkpoints in
`checkpoints/baseline_restormer_10ep/`, artifacts in
`results/restormer_baseline_10ep/`.

| GoPro test split (1111 images, 256px center crop) | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Restormer, 10 epochs (superseded) | 25.468 +/- 4.097 dB | 0.7586 +/- 0.1474 | 0.2834 +/- 0.1496 |

Validation PSNR was still climbing steeply when the 10-epoch cosine schedule
expired, so the run ended because the schedule ended, not because the model
converged. It is a pipeline-validation datapoint only. **Do not cite it as a
baseline or as a comparison point for any ablation.**

### 5.3 Ablation study — in progress

| # | Preset | Status |
|---|---|---|
| A | `baseline_restormer` | **complete, validated** (§5.1) |
| B | `seecoder` | **running** — 60 epochs, no results yet, none may be cited |
| C | `seecoder_diffae` | not started |
| D | `seecoder_diffae_lcr` | not started (the novelty under test) |
| E | `seecoder_diffae_lcr_flow` | not started |
| F | `full_srrt` | not started |

Ablation B was gated before launch by the zero-gate check (§1.2, exact),
the 8-point sanity test and the overfit test, all re-run against the `seecoder`
configuration after the injection fix in §4.
