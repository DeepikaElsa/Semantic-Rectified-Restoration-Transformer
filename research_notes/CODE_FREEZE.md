# SRRT — Code Freeze Audit

**Date:** 2026-08-28
**Machine:** Windows 11, RTX 4070 Laptop (8187 MiB), Python 3.11.9, torch 2.11.0+cu128 (CUDA 12.8)
**Conducted while** Ablation 1 (`seecoder`) was training. Nothing in the active
experiment was touched: no checkpoint, no CSV, no optimizer/scheduler state, no
config, no package installed. Every check below ran on **CPU**.

> **CPU isolation note.** `CUDA_VISIBLE_DEVICES=""` does **not** hide the GPU on
> this Windows/torch build — `torch.cuda.is_available()` stays `True`,
> `device_count()` reports 0, and a CUDA allocation still succeeds. Only
> `CUDA_VISIBLE_DEVICES=-1` genuinely disables it. All results recorded here were
> re-run under `-1`; an earlier suite run under `""` did touch the GPU and is not
> counted as a CPU-only result.

---

## A. Completed components

| Component | State |
|---|---|
| Restormer backbone (MDTA channel attention, GDFN, encoder/bottleneck/decoder/refinement) | complete, 26,126,644 params |
| SRRT modules — SeeCoder, DiffAE, LCR, Rectified Flow, SRA | complete, all six ablation presets construct and train |
| Zero-gated integration (`inject` + `gamma_latent`, `sra.gamma`, `flow.proj_out`) | complete; bitwise-exact no-op at init |
| Ablation framework (`utils/ablation.py`, `--ablation`) | complete, 6 presets |
| Combined loss (L1 + perceptual + flow + consistency) | complete, composition verified by exact arithmetic |
| GoPro dataset + deterministic evaluation | complete, 2103 train / 1111 test pairs |
| Training loop — AMP, grad accumulation, clipping, checkpoint/resume, CSV + TensorBoard | complete |
| `evaluate.py` — PSNR / SSIM / LPIPS, per-image CSV + summary JSON | complete |
| Analysis + figure scripts (13) | complete, all verified runnable |
| Test suite | 171 tests, all passing |

### Analysis infrastructure added during the freeze

| Script | Purpose |
|---|---|
| `scripts/audit_code_freeze.py` | this audit, re-runnable, exits non-zero on failure |
| `scripts/compare_paired_metrics.py` | paired per-image Δ with mean/median/std/95% CI |
| `scripts/plot_convergence_comparison.py` | multi-run convergence overlay |
| `scripts/efficiency_table.py` | cost table from measured artifacts only |
| `scripts/run_ablation1_analysis.py` | 10-step post-training driver, guarded on epoch count |

---

## B. Tests passed

| Check | Result |
|---|---|
| Full test suite, CPU-only (`CUDA_VISIBLE_DEVICES=-1`) | **171 / 171 passed** (41 s) |
| Code-freeze audit (`scripts/audit_code_freeze.py`) | **106 / 106 checks passed** |
| Every project module imports | **81 / 81**, 0 failures |
| Doc/config consistency tests | 22 passed |

### Audit coverage, per ablation (all six)

Construction · forward · output shape == input shape · finite output · backward ·
finite gradients · optimizer updates parameters · device placement ·
bitwise equality to pure Restormer at init · zero gates · aux-loss presence ·
loss composition · module connectivity.

**Zero-gate result — all six configurations:** `max|Δ| = 0.000e+00` against the
bare Restormer with shared backbone weights. Gates at init:
`gamma_latent = 0`, `sra.gamma = 0`, `flow.proj_out.weight = 0`.

### Ablation enable matrix (machine-readable: `research/tables/code_freeze_audit.json`)

| Configuration | SeeCoder | DiffAE | LCR | Flow | SRA |
|---|:--:|:--:|:--:|:--:|:--:|
| `baseline_restormer` | False | False | False | False | False |
| `seecoder` | True | False | False | False | False |
| `seecoder_diffae` | True | True | False | False | False |
| `seecoder_diffae_lcr` | True | True | True | False | False |
| `seecoder_diffae_lcr_flow` | True | True | True | True | False |
| `full_srrt` | True | True | True | True | True |

Matches the intended matrix exactly; the audit fails if any row deviates.

### Gated vs disconnected

Two independent mechanisms, because **one is not sufficient**:

1. `scripts/zero_gate_check.py` — output identical to Restormer at init.
2. `tests/test_srrt.py::test_enabled_modules_are_actually_connected` — every
   enabled module receives non-zero gradient **from the full training objective**
   (including `flow_loss` / `lcr_loss`, not the pixel term alone).

A gated module and a module wired to nothing both score `0.000e+00` on (1). Only
(2) separates them. This distinction is not decorative: SeeCoder was in fact
disconnected until 2026-08-28 (`VALIDATION_STATUS.md` §4). **Do not remove the
connectivity test.**

### Loss integrity

```
L_total = λ_1 · L1  +  λ_p · L_perceptual  +  λ_f · L_flow  +  λ_c · L_consistency
          λ_1 = 1.0    λ_p = 0.1 (VGG19)     λ_f = 0.1        λ_c = 0.1
```

`L_flow` is added **iff** Rectified Flow is enabled; `L_consistency` **iff** LCR
is enabled (`losses/combined_loss.py` adds a term only when its tensor is not
None, and `train.py` passes `out.get("flow_loss")` / `out.get("lcr_loss")`).
The audit asserts, per ablation, that each auxiliary loss is present exactly when
its module is enabled, and that `total` equals the sum of weighted components.

---

## C. Known limitations (code)

1. **`device.gpu` in `configs/srrt_base.yaml` is never read.** `train.py` uses
   `torch.device("cuda" if torch.cuda.is_available() else "cpu")`, so setting
   `gpu: 1` would silently still use device 0. Harmless on this single-GPU
   machine, misleading anywhere else. **Deferred**: fixing it edits `train.py`,
   which the active run would pick up on resume.
2. **`diffae.semantic_head` receives no gradient** — 131,584 params (9.7% of
   DiffAE) computed every forward pass and never entering the loss. Its output
   (`damage_semantic`) is surfaced only under `return_features=True`, for
   visualization. It inflates the reported parameter count for ablations C–F and
   costs a wasted Linear→GELU→Linear per step. Reported by the audit under
   `untrained_tensors`. **Decision needed before Ablation 2**, not before now.
3. **GFLOPs cannot be measured** — neither `thop` nor `fvcore` is installed, and
   nothing was installed during training. Every table shows `unavailable`;
   nothing is estimated.
4. **`scripts/plot_training.py` is superseded** by `plot_training_curves.py`. Kept
   deliberately so the archived 10-epoch figures remain reproducible.
5. **Only GoPro / deblurring.** Other datasets raise `NotImplementedError`.

## D. Known experimental limitations

1. **Validation and test are the same split.** `dataset.val.split = test`, so
   best-checkpoint selection and the reported metrics use the same 1111 images.
   **There is no held-out test set.** The paper must not describe it as one —
   these are validation numbers, possibly mildly optimistic through selection.
2. **Run-to-run variance is unmeasured.** One training run per arm. No seed-repeat
   study exists, so a small ablation delta cannot be called an improvement, and a
   statistically significant per-image difference may still be seed noise.
3. **Confidence intervals are over images, not runs.** The 95% CI from
   `compare_paired_metrics.py` bounds sampling over test images for one fixed pair
   of checkpoints. It is **not** a training-run variance estimate.
4. **Not comparable to published Restormer.** 15,720 optimizer steps at a fixed
   128 px crop, scored on 256 px centre crops, versus ~300k iterations on full
   images with a progressive 128→384 schedule. The baseline's 28.937 dB is **not**
   a Restormer reproduction.
5. **Train/inference asymmetry in the flow path** (`design_decisions.md` D9):
   training injects the supervised endpoint z₁, inference the ODE solution.
6. **Loss weights untuned** — `(1.0, 0.1, 0.1, 0.1)` are spec defaults.
7. **Convergence curves across runs are not paired trajectories.** Same seed and
   config, but different architectures consume the RNG stream differently.

## E. Remaining experiments

| # | Preset | State |
|---|---|---|
| A | `baseline_restormer` | ✅ complete — 60 epochs, best epoch 57 |
| B | `seecoder` | 🔄 **training now** — no result may be cited |
| C | `seecoder_diffae` | not started |
| D | `seecoder_diffae_lcr` | not started — the novelty under test |
| E | `seecoder_diffae_lcr_flow` | not started |
| F | `full_srrt` | not started |

Optional but recommended: a **seed-repeat of the baseline** (2–3 seeds). Without
it, limitation D.2 caps what any ablation result can claim.

## F. Commands to reproduce

```bash
# gates (run before any training run)
python scripts/zero_gate_check.py --size 128 --json results/<exp>/zero_gate.json
python scripts/sanity_check.py  --ablation <preset> --iters 15 --patch 128
python scripts/overfit_test.py  --ablation <preset> --num-images 16 --epochs 40

# training  (--epochs retunes scheduler T_max; train.py aborts if T_max != epochs)
python train.py --ablation baseline_restormer       --epochs 60
python train.py --ablation seecoder                 --epochs 60
python train.py --ablation seecoder_diffae          --epochs 60
python train.py --ablation seecoder_diffae_lcr      --epochs 60
python train.py --ablation seecoder_diffae_lcr_flow --epochs 60
python train.py --ablation full_srrt                --epochs 60
# resume:  --resume checkpoints/<preset>/latest.pth
# on Windows, run scripts/keep_awake.py alongside: Modern Standby kills long runs

# evaluation (identical protocol for every arm)
python evaluate.py --checkpoint checkpoints/<preset>/best_psnr.pth \
                   --split test --out-dir results/<name>

# full post-training analysis for Ablation 1
python scripts/run_ablation1_analysis.py          # --dry-run to preview

# audit (safe beside a live run)
CUDA_VISIBLE_DEVICES=-1 python scripts/audit_code_freeze.py
CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/ -q
```

## G. Paper figures — READY

| Figure | Status |
|---|---|
| 3 — Zero-gated module integration (measured panel) | **READY** — `results/ablation_seecoder/zero_gate.json` + gradient trace |
| 4 — Training convergence (baseline) | **READY** — `plot_training_curves.py` |
| 9 — Ablation comparison (baseline rung only) | **READY as a partial ladder** — unrun rungs render as hatched "not run", never interpolated |
| 10 — Efficiency (cost half, GFLOPs excepted) | **READY** — params measured for all six presets |

Figures 1, 2 and the schematic halves of 3 are **manual diagrams**, drawable now.

## H. Paper figures — BLOCKED

| Figure | Blocked by |
|---|---|
| 4 (overlay) | Ablation 1 finishing |
| 5 — Qualitative comparison | Ablation 1 finishing (script verified, baseline-only today) |
| 6 — Error maps | Ablation 1 finishing |
| 7 — Frequency analysis | Ablation 1 finishing |
| 8 — Feature/attention (SeeCoder panels) | Ablation 1 finishing |
| 9 (full ladder) | Ablations 2–5 — four runs not started |
| 10 (accuracy axis) | same as 9 |
| 10 (GFLOPs cell) | no FLOP counter installed; must not be estimated |

All nine visualization scripts were executed on CPU against the real baseline
checkpoint during this audit and produced output with provenance sidecars.

---

## CODE FREEZE STATUS: **READY**

All planned coding/infrastructure components are complete. Remaining work is
experimental training, evaluation, statistical analysis, visualization, and paper
writing.

### Fixed during this audit

| Issue | Impact |
|---|---|
| `make_comparison_table.py` missing `sys.path` bootstrap | script could not run at all as documented (`ModuleNotFoundError`) |
| `find_summary()` matched only `results/<preset_name>/` | the measured baseline rendered as "not run" in **three** table/figure scripts; fixed with aliases + a scan of each `summary.json`'s own `experiment` field, excluding archived dirs |
| `generate_qualitative_grid.py` error-column loop unguarded | `IndexError` whenever `--with-error` was omitted (a defect in this session's own edit) |
| `audit_code_freeze.py` compared outputs from two different random inputs | made all six zero-gate rows falsely fail, baseline included |

### Deferred (not blocking, must not be done while training runs)

1. Honour or remove `device.gpu` in the config — edits `train.py`.
2. Decide DiffAE `semantic_head`: keep (visualization only), drop, or attach a
   reconstruction/semantic loss. Must be settled **before Ablation 2**, since it
   changes ablation C's parameter count.
3. Install `thop` to populate GFLOPs, then re-run `scripts/profile_models.py`
   while the GPU is free.
