# SRRT — Semantic Rectified Restoration Transformer

**This repository contains a *proposed* research architecture. No experimental
claim in it is supported yet: the model has not been trained to convergence, and
no PSNR/SSIM/LPIPS result should be quoted from this code until you run the
experiments yourself.**

---

## 1. Research motivation & problem statement

Image restoration inverts an unknown degradation:

```
y = D(x) + n          observed y, unknown clean x, degradation D, noise n
```

The problem is ill-posed — many `x` explain the same `y`. Transformer backbones
such as Restormer restore effectively from low-level evidence alone, but they
have no explicit representation of *what* was damaged or of the *semantic
content* being reconstructed.

SRRT proposes to add that: encode visual context and degradation cues
separately, fuse them into a latent that is refined for consistency with the
clean-image latent, transport that latent along a learned straight-path ODE, and
let the decoder's own output be rectified by a self-derived residual.

**The primary proposed novelty is LCR (Latent Consistency Refinement).** The
overall contribution is the combination of visual-context encoding,
damage-aware encoding, latent consistency refinement, rectified-flow transport,
efficient transformer restoration, and self-rectifying residual refinement.

## 2. Proposed architecture

```
                    y (degraded)
                         |
            +------------+------------+
            |                         |
        SeeCoder                   DiffAE
     visual context          damage-aware semantic
            |                         |
            +------------+------------+
                         |
                        LCR                    <-- proposed novelty
            Latent Consistency Refinement
                         |
                  Rectified Flow
              latent-space ODE transport
                         |
                     Restormer
              (latent injected at bottleneck)
                         |
                        SRA
              Self-Rectifying Attention
                         |
                   x_hat (restored)
```

Full equations: [`docs/mathematical_formulation.md`](docs/mathematical_formulation.md).
Worked numbers: [`docs/numerical_example.md`](docs/numerical_example.md).
Every deviation and ambiguity: [`docs/design_decisions.md`](docs/design_decisions.md).

## 3. Implementation status

Status is tracked in four distinct levels. **Nothing has reached level 4 (experimentally validated).** See `research_notes/VALIDATION_STATUS.md` and `research_notes/BASELINE_RESULTS.md`.

| Component | Implemented | Unit-tested | Trained | Validated |
|---|:--:|:--:|:--:|:--:|
| LayerNorm / to_3d / to_4d | yes | yes | no | no |
| MDTA (channel attention) | yes | yes | no | no |
| GDFN | yes | yes | no | no |
| TransformerBlock / Stage | yes | yes | no | no |
| Encoder / Bottleneck / Decoder | yes | yes | no | no |
| Restormer backbone | yes | yes | **10 ep (not converged)** | no |
| GoPro dataset + paired transforms | yes | yes | n/a | n/a |
| SeeCoder | yes | yes | no | no |
| DiffAE | yes | yes | no | no |
| **LCR** | yes | yes | **no** | **no** |
| Rectified Flow | yes | yes | no | no |
| SRA | yes | yes | no | no |
| Full SRRT integration | yes | yes | no | no |
| Combined loss | yes | yes | no | no |
| Training loop | yes | **yes (10-epoch run)** | n/a | n/a |
| Evaluation (PSNR/SSIM/LPIPS) | yes | **yes (1111 images)** | n/a | n/a |
| Ablation framework | yes | yes | no | no |
| Datasets beyond GoPro | no | — | — | — |

## 4. Repository layout

```
srrt/
├── configs/srrt_base.yaml          all hyperparameters (nothing hardcoded)
├── data/
│   ├── base.py                     PairedRestorationDataset + pairing validation
│   ├── transforms.py               paired crops / flips / rotations
│   └── datasets.py                 GoPro + registry + dataloader builders
├── models/
│   ├── common/                     layers, attention (MDTA), feedforward (GDFN), blocks
│   ├── backbone/                   encoder, bottleneck, decoder, restormer
│   ├── modules/                    seecoder, diffae, lcr, rectified_flow, sra
│   └── srrt.py                     full integration
├── losses/combined_loss.py         L1 + perceptual + flow + LCR
├── utils/                          seed.py (repro/env), ablation.py (presets)
├── scripts/                        run_baseline.py, run_srrt.py, make_comparison_table.py
├── tests/                          149 tests
├── docs/                           mathematical_formulation, numerical_example, design_decisions
├── train.py
├── evaluate.py
└── requirements.txt
```

## 5. Installation

```bash
python -m venv .venv
.venv\Scripts\activate                # Windows
pip install -r requirements.txt
```

Verify the GPU:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

A system CUDA version newer than the PyTorch CUDA runtime is fine (drivers are
backward compatible). This project was developed against system CUDA 12.9 with
`torch 2.11.0+cu128`.

## 6. Dataset setup

GoPro, already in place:

```
data/Gopro/
├── train/{blur,sharp}/       2103 pairs
└── test/{blur,sharp}/        1111 pairs
```

`blur` → input, `sharp` → target. Filenames must correspond; the loader
validates this per-file and raises rather than training against misaligned
targets.

To add a dataset, subclass `PairedRestorationDataset` and add one
`DATASET_REGISTRY` entry — `train.py`/`evaluate.py` need no changes. Planned:
Rain100H (deraining), RESIDE (dehazing), LOL-v2 (low-light), SIDD (denoising),
DIV2K (super-resolution). They currently raise `NotImplementedError` by design.

## 7. Commands

```bash
# --- dataset test ---
python -m pytest tests/test_dataset.py -v

# --- Restormer backbone tests (incl. the MDTA HWxHW regression guard) ---
python -m pytest tests/test_attention.py tests/test_restormer.py -v -s

# --- SRRT forward test (all 6 ablation stages) ---
python -m pytest tests/test_srrt.py -v -s

# --- training sanity checks (forward / backward / optimizer / overfit) ---
python -m pytest tests/test_training_sanity.py -v -s

# --- whole suite ---
python -m pytest tests/ -q

# --- VALIDATION GATES (run in this order before trusting any result) ---
python scripts/audit_pipeline.py                       # 20-item pre-training audit
python scripts/zero_gate_check.py                      # SRRT == Restormer at init
python scripts/sanity_check.py                         # forward/backward/optimizer/loss
python scripts/overfit_test.py                         # can the pipeline learn at all?
python scripts/measure_throughput.py                   # epoch time + VRAM projection

# --- training ---
python train.py                                        # config default (baseline)
python train.py --ablation full_srrt
python train.py --max-steps 20 --max-val-batches 5     # quick smoke test
python train.py --resume checkpoints/full_srrt/latest.pth

# --- evaluation ---
python evaluate.py --checkpoint checkpoints/baseline_restormer/best_psnr.pth
python evaluate.py --checkpoint <ckpt> --full-images    # protocol-comparable, slower

# --- ablation sweep + comparison table ---
python scripts/run_baseline.py
python scripts/run_srrt.py --all
python scripts/make_comparison_table.py --markdown results/comparison.md

# --- TensorBoard ---
tensorboard --logdir logs
```

## 8. Configuration & ablations

Everything lives in `configs/srrt_base.yaml`. Modules toggle via
`model.modules.<name>.enabled`; `utils/ablation.py` provides the six presets:

| Preset | Modules | Params |
|---|---|---|
| `baseline_restormer` | Restormer only | 26,126,644 |
| `seecoder` | + SeeCoder | 26,462,024 |
| `seecoder_diffae` | + DiffAE | 27,823,016 |
| `seecoder_diffae_lcr` | + LCR | 33,098,709 |
| `seecoder_diffae_lcr_flow` | + Rectified Flow | 37,166,677 |
| `full_srrt` | + SRA | 37,235,130 |

Each run writes to its own `checkpoints/<name>/`, `logs/<name>/`,
`outputs/<name>/`, `results/<name>/`, so stages never overwrite each other.

**Why this ladder is trustworthy:** with all modules disabled SRRT is
*numerically identical* to the bare Restormer, and every module joins through a
zero-initialized gate. So each stage starts from the baseline and any measured
difference is attributable to what that module learned — not to incidental
perturbation. Both properties are asserted in the test suite.

## 9. Hardware requirements (measured, RTX 4070 Laptop 8 GB)

Inference, 256×256, `dim=48`:

| Component | Peak VRAM | Time |
|---|---|---|
| MDTA alone | 104 MB | 0.16 s |
| Encoder | 255 MB | 0.04 s |
| Encoder+Bottleneck+Decoder | 491 MB | 0.22 s |
| Full Restormer | 494 MB | 0.14 s |
| Full SRRT (all modules) | ~6.9 GB observed in-loop | — |

Training, forward+backward+AMP+perceptual loss:

| Config | Peak VRAM |
|---|---|
| B=1, 128² | 1.68 GB |
| **B=2, 128² (default)** | **3.36 GB** |
| B=4, 128² | 6.43 GB |
| B=1, 256² | 6.2 GB |
| B=2, 256² | OOM |

`device.max_memory_fraction: 0.85` caps the allocator so an over-large config
raises a clean `OutOfMemoryError` instead of silently spilling into Windows
shared GPU memory (system RAM) — see `docs/design_decisions.md` D3.

## 10. Known limitations

1. **Nothing is trained.** No quality claim can be made from this repository.
2. **A full 200-epoch run takes ~34 h on this GPU.** Measured steady state is
   2.7-3.0 it/s (333-372 ms/iter) at batch 2 / 128px / AMP: ~6.5 min of training
   per epoch over 1051 iterations, plus ~3.7 min if validating on all 1111 test
   images (198 ms/image at 256px). `training.val_max_batches` caps per-epoch
   validation to a subset to keep epochs cheap; measure the full test set once at
   the end with `evaluate.py`. See `results/throughput.json` and
   `scripts/measure_throughput.py`.
3. **Center-crop validation is not protocol-comparable** to published GoPro
   numbers (which use full images). Use `--full-images` for comparable figures.
4. **Train/inference mismatch in the flow path.** During training the backbone
   receives the supervised endpoint `z1`; at inference it receives the ODE
   solution (D9). Exposure bias is unaddressed pending experiments.
5. **Loss weights are untuned** spec defaults.
6. **No progressive patch-size schedule** (the paper's 128→384 ramp).
7. **Only GoPro** is implemented.
8. **DiffAE does not provably isolate degradation** — it is a learned
   damage-aware representation only (D12).
9. **Not a git repository**, so `git_hash` is recorded as `null`. Run
   `git init` to make checkpoints traceable to commits.
10. **`num_workers: 4`** by default on Windows; set to 0 if you hit DataLoader
    spawn issues.

## 11. Reproducibility

`utils/seed.py` seeds Python, NumPy and PyTorch (CPU + all CUDA devices), with
optional deterministic cuDNN via `experiment.deterministic`. Every checkpoint
embeds the full config plus an environment snapshot (Python, torch, CUDA, GPU
name, VRAM, git hash), and `evaluate.py` writes `config.yaml`, `metrics.csv` and
`summary.json` next to each result.

## 12. Citation & honesty policy

Restormer is a published architecture (Zamir et al., *Restormer: Efficient
Transformer for High-Resolution Image Restoration*, CVPR 2022) and is used here
as the baseline backbone. Rectified Flow follows Liu et al., *Flow Straight and
Fast*, ICLR 2023.

SeeCoder, DiffAE, LCR and SRA as implemented here are **proposed SRRT
components**, not established methods. They must be mathematically defined
(done — see the docs) *and* experimentally validated (not done). Do not cite them
as prior art, and do not report any number from this repository that you have not
produced yourself with a real training run.
