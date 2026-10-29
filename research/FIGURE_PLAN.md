# SRRT — Paper Figure Plan

**Updated:** 2026-08-28

**Status of the experimental record at time of writing:**

| Experiment | State |
|---|---|
| Restormer baseline (`baseline_restormer`) | ✅ **complete and evaluated** — 60 epochs, best epoch 57, PSNR 28.937 / SSIM 0.8661 / LPIPS 0.1356 on 1111 test images |
| Ablation 1 `seecoder` | 🔄 **training in progress** — no evaluation exists, no number may be cited |
| Ablations 2–5 (`seecoder_diffae` … `full_srrt`) | ⛔ **not started** |

> **No SRRT module has produced a measured result.** No figure in this document
> may be used to claim SRRT improves on Restormer. Figures 5–8 are baseline-only
> until Ablation 1 finishes, at which point they gain a second column.

Availability legend:

| Mark | Meaning |
|---|---|
| ✅ **available** | can be generated right now from measured data |
| 🟡 **partial** | generatable, but baseline-only or from an unfinished run |
| ⛔ **blocked** | requires an experiment that has not been run |
| ✏️ **manual** | a diagram to be drawn, not produced from data |

---

## Figure 1 — Overall SRRT architecture

- **Purpose:** the data path — degraded image → Restormer encoder → bottleneck latent, with SeeCoder/DiffAE/LCR/Rectified-Flow acting on the latent and SRA on the refined features.
- **Required experiment:** none (schematic).
- **Required data:** none. Source of truth is `models/srrt.py::SRRT.forward` and `docs/design_decisions.md`.
- **Availability:** ✏️ **manual** — to be drawn (TikZ/Inkscape).
- **Limitations:** a schematic proves nothing about behaviour. It must depict the *current* wiring, including that SeeCoder/DiffAE reach the bottleneck through `inject` + `gamma_latent` even when LCR is absent — that path did not exist before 2026-08-28 (see `research_notes/VALIDATION_STATUS.md` §4).

## Figure 2 — Mathematical pipeline / latent restoration process

- **Purpose:** z₀ (degraded latent) → rectified-flow transport → ẑ₁, and the LCR consistency objective, as equations aligned to the code.
- **Required experiment:** none for the schematic; a *measured* flow-loss trajectory needs a trained flow model (Ablation 4).
- **Required data:** none for the schematic.
- **Availability:** ✏️ **manual** for the schematic; ⛔ **blocked** for any measured overlay.
- **Limitations:** must show the train/inference asymmetry (`docs/design_decisions.md` D9) — training injects the supervised endpoint z₁, inference injects the ODE solution. Hiding that would misrepresent what the model does at test time.

## Figure 3 — Module integration and zero-gated initialization

- **Purpose:** how each module attaches as a *zero-initialized additive* branch, so untrained SRRT **is** Restormer, and how the gate opens during training.
- **Required experiment:** none for the schematic; the measured panel uses `results/ablation_seecoder/zero_gate.json` (already written).
- **Required data:** ✅ measured — zero-gate table (all six configurations, max/mean/relative error all `0.000e+00`), plus the gradient-flow trace (step 1: only `gamma_latent` receives gradient, 5.27e-05; step 2+: all 44 SeeCoder tensors do).
- **Availability:** ✏️ **manual** for the wiring diagram; ✅ **available** for the measured panel.
- **Limitations:** **identical output at init does not prove a module is connected.** A gated module and a module wired to nothing both score exactly 0. This figure should show *both* facts — the zero-gate equality and the gradient evidence — because the first alone concealed a dead SeeCoder path. Connectivity is enforced by `tests/test_srrt.py::test_enabled_modules_are_actually_connected`.

## Figure 4 — Training convergence

- **Purpose:** show the baseline trained to a schedule-consistent optimum, and overlay each ablation.
- **Required experiment:** baseline ✅; Ablation 1 in progress.
- **Required data:** `logs/<experiment>/train_log.csv`.
- **Availability:** ✅ **available** for the baseline; 🟡 **partial** for the overlay until `seecoder` finishes (the script plots a live run as a shorter line).
- **Commands:**
  ```bash
  python scripts/plot_training_curves.py --experiment baseline_restormer --individual
  python scripts/plot_convergence_comparison.py \
      --run "Restormer=baseline_restormer" \
      --run "Restormer + SeeCoder=seecoder" \
      --out-dir research/figures --stem ablation_seecoder_convergence
  ```
- **Limitations:** ⚠ **these are independent runs, not a paired trajectory.** Same seed and configuration, but different architectures consume the RNG stream differently from the first step, and `deterministic: false`. Run-to-run variance is **unmeasured** — no seed-repeat study exists — so a gap between curves at any epoch is *not* a controlled measurement of the module's effect. The figure carries this caption on its face.

## Figure 5 — Qualitative restoration comparison

- **Purpose:** Input | Ground Truth | Restormer | +SeeCoder, plus a variant with an Error column; zoomed crops on detailed regions.
- **Required experiment:** baseline ✅; +SeeCoder column pending Ablation 1.
- **Required data:** test images + one checkpoint per model shown.
- **Availability:** 🟡 **partial** — baseline-only today.
- **Output:** `research/figures/ablation_seecoder_qualitative.{png,pdf}` and `…_qualitative_error.{png,pdf}`.
- **Commands:**
  ```bash
  python scripts/generate_qualitative_grid.py \
      --model "Restormer=checkpoints/baseline_restormer/best_psnr.pth" \
      --model "Restormer + SeeCoder=checkpoints/seecoder/best_psnr.pth" \
      --num_samples 6 --seed 42 --output_dir research/figures \
      --stem ablation_seecoder_qualitative
  # add --with-error for the Error-column variant
  ```
- **Limitations:** sample selection is by `utils.research.select_image_ids`, which sees **no model and no metric**, and the zoom window is chosen from ground-truth gradient energy alone — so neither can be steered toward images that flatter one model. Selection is nested in `num_samples`, so seed 42 with n=6 is a prefix of seed 42 with n=8, keeping every figure on the same images. Six images cannot represent 1111; qualitative panels illustrate, they do not measure.

## Figure 6 — Error-map comparison

- **Purpose:** where error concentrates, using E(x,y) = ⅓ Σ_c |I_pred,c − I_gt,c|.
- **Required experiment:** baseline ✅; comparison pending Ablation 1.
- **Required data:** same images for all models, one shared colour scale.
- **Availability:** 🟡 **partial**.
- **Command:**
  ```bash
  python scripts/visualize_error_maps.py \
      --model "Restormer=checkpoints/baseline_restormer/best_psnr.pth" \
      --model "Restormer + SeeCoder=checkpoints/seecoder/best_psnr.pth" \
      --num_samples 6 --seed 42 --modes abs_mean,squared_mean \
      --shared-scale-across-images
  ```
- **Limitations:** models are **never** independently rescaled — one `vmax` spans all models (and, with `--shared-scale-across-images`, all images), so a darker panel means less error rather than a different colour axis. Per-model normalization here would manufacture a visible "improvement" from nothing.

## Figure 7 — Frequency-domain analysis

- **Purpose:** log(1 + |FFT₂(I)|) spectra and azimuthally averaged radial profiles for Input / GT / Restormer / +SeeCoder.
- **Required experiment:** baseline ✅; comparison pending Ablation 1.
- **Required data:** same images, all models.
- **Availability:** 🟡 **partial**.
- **Command:**
  ```bash
  python scripts/visualize_frequency.py \
      --model "Restormer=checkpoints/baseline_restormer/best_psnr.pth" \
      --model "Restormer + SeeCoder=checkpoints/seecoder/best_psnr.pth" \
      --num_samples 6 --seed 42
  ```
- **Limitations:** ⚠ **more high-frequency energy is not automatically better restoration.** Hallucinated texture, ringing and noise all raise the high-frequency band, and the magnitude spectrum discards phase entirely — an image can match GT spectrally and still be visibly wrong. This figure is **exploratory evidence only**; quality claims rest on PSNR/SSIM/LPIPS.

## Figure 8 — Feature and attention analysis

- **Purpose:** channel-mean activations A(x,y) = (1/C) Σ_c |F_c(x,y)| across encoder / bottleneck / decoder, SeeCoder context features, and MDTA channel-attention statistics.
- **Required experiment:** Restormer stages + MDTA ✅ (baseline); SeeCoder panels pending Ablation 1.
- **Required data:** one checkpoint; SeeCoder panels need that module enabled.
- **Availability:** 🟡 **partial**.
- **Commands:**
  ```bash
  python scripts/visualize_features.py  --checkpoint checkpoints/seecoder/best_psnr.pth
  python scripts/visualize_attention.py --checkpoint checkpoints/baseline_restormer/best_psnr.pth
  ```
- **Limitations:** ⚠ **a striking feature map is not evidence of semantic understanding.** Channel-mean activation is an aggregate magnitude, not a semantic readout; high activation marks where the network puts energy, not what it "recognises". Attention maps do not establish causality either. Causal claims come only from the ablation ladder.

## Figure 9 — Ablation study

- **Purpose:** PSNR / SSIM / LPIPS across the six-rung ladder (Restormer → +SeeCoder → +DiffAE → +LCR → +Rectified Flow → +SRA).
- **Required experiment:** **five training runs; one is in progress, four not started.**
- **Required data:** `results/<ablation>/summary.json` for every rung, all under one evaluation protocol.
- **Availability:** ⛔ **blocked** — one measured rung. The script renders unrun rungs as hatched "not run" and never interpolates.
- **Commands:**
  ```bash
  python scripts/make_comparison_table.py --markdown research/tables/ablation_main.md
  python scripts/compare_ablation_results.py --errorbars
  python scripts/compare_paired_metrics.py \
      --baseline results/restormer_baseline --ablation results/ablation_seecoder
  ```
- **Limitations:** error bars are the per-image **standard deviation**, not a confidence interval and not seed variance. The paired per-image comparison reports a 95% CI, but that CI is over **test images for one fixed pair of checkpoints** — it is **not** a training-run variance estimate, and with a single run per arm a statistically significant difference may still be seed noise. Validation and test are the same split, so both checkpoints were selected on the data they are scored on.

## Figure 10 — Computational efficiency

- **Purpose:** accuracy against cost — parameters, train time, median epoch time, peak VRAM, inference latency, GFLOPs.
- **Required experiment:** cost measurable per configuration; the accuracy axis needs the ablation runs.
- **Required data:** `logs/<run>/train_log.csv`, `results/<run>/summary.json`, optional `research/tables/profile.json`.
- **Availability:** 🟡 **partial** — baseline fully measured; SeeCoder cost partly measured from the live log, accuracy pending.
- **Command:**
  ```bash
  python scripts/efficiency_table.py \
      --run "Restormer=baseline_restormer=results/restormer_baseline" \
      --run "Restormer + SeeCoder=seecoder=results/ablation_seecoder"
  ```
- **Limitations:** **GFLOPs are `unavailable` and are never estimated** — neither `thop` nor `fvcore` is installed in this environment, and a guessed FLOP count would be indistinguishable in the table from a measured one. To populate it, install a counter and run `scripts/profile_models.py` **while the GPU is free**. Wall-clock timings come from one laptop GPU under WDDM; they are comparable to each other (same machine, same configuration) but are not a controlled benchmark. Two baseline epochs spanning Windows Modern Standby (17,303 s and 5,515 s) are excluded as sleep, not compute.

---

## Summary of what is blocked and why

| Figure | Blocked by |
|---|---|
| 1, 2, 3 (diagrams) | nothing — manual, can be drawn now |
| 3 (measured panel) | nothing — zero-gate + gradient data already written |
| 4 (baseline) | nothing — ✅ available |
| 4 (overlay), 5, 6, 7, 8 | Ablation 1 finishing |
| 9 | four further ablation runs + evaluation |
| 10 (accuracy axis) | same as 9 |
| 10 (GFLOPs cell) | no FLOP counter installed; must not be estimated |

**Nothing in this plan authorizes starting Ablation 2.** Ablation 1 must finish
and be read first; whether DiffAE runs at all is a decision that follows those
results.

## Provenance

Every script writes a `<figure>.json` sidecar next to its output recording:
experiment, checkpoint, epoch, dataset, split, image IDs, seed, resolution,
crop mode, normalization, git hash, and the exact argv. A figure without a
sidecar was not produced by this pipeline and must not enter the paper.
