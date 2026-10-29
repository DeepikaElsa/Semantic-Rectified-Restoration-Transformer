# SRRT Research Notes

## Phase 1/2 — Restormer baseline rebuild (2026-08-11)

### Critical bug found and fixed: MDTA was computing HW x HW spatial attention

`models/attention.py` looked correct (channel/transposed attention with
normalized Q/K, learnable temperature, depthwise QKV conv) but the matmul
order was wrong:

- Q, K, V were reshaped to `(B, heads, head_dim, H*W)` — correct.
- The forward pass then computed `attn = matmul(q.transpose(-2,-1), k)`,
  i.e. `(B,heads,HW,head_dim) @ (B,heads,head_dim,HW) -> (B,heads,HW,HW)`.
  This is spatial attention (HW x HW), not channel attention.
- At H=W=256, dim=48, this produced ~65536 x 65536 matrices — the ~34GB
  allocation that previously crashed the machine.

**Fix**: swap the matmul order to `attn = matmul(q, k.transpose(-2,-1))`,
giving `(B,heads,head_dim,HW) @ (B,heads,HW,head_dim) -> (B,heads,head_dim,head_dim)`,
and normalize Q/K along the HW dim (`dim=-1`) rather than the channel dim,
matching the original Restormer formulation. Output is then
`attn @ v -> (B,heads,head_dim,HW)`, reshaped directly back to `(B,C,H,W)`.

Verified empirically: full Restormer forward at B=1, C=3, H=W=256 with the
default config (dim=48, encoder_blocks=[4,6,6,8], decoder_blocks=[6,6,4],
heads=[1,2,4,8]) now uses ~492MB peak GPU memory and completes in
~0.15-0.5s on the dev GPU. See `tests/test_attention.py::test_mdta_at_realistic_resolution_stays_cheap`
and `tests/test_restormer.py::test_restormer_realistic_image_tensor`.

`tests/_utils.py::assert_no_quadratic_attention` guards against this
regressing again: it monkeypatches `torch.matmul` for the duration of a
forward pass and fails if any intermediate matrix's last two dims both
exceed the expected channel-attention bound.

### Design decisions / assumptions

- **LayerNorm**: `models/blocks.py` previously defined its own ad-hoc
  `LayerNorm2d` (a thin wrapper around `nn.LayerNorm`). This was replaced
  with the already-tested `models.layers.LayerNorm` (BiasFree/WithBias,
  `to_3d`/`to_4d`) to avoid two parallel norm implementations. Default is
  `"WithBias"`, matching `models/layers.py`'s own default and the variant
  the original Restormer paper uses for most restoration tasks (e.g.
  motion deblurring).
- **Decoder level 1 has no channel-reduction conv**, staying at `dim*2`
  (96 channels for dim=48) after concatenating with the level-1 skip
  connection. This matches the original Restormer, which intentionally
  keeps extra capacity at the highest resolution rather than reducing it
  like levels 2 and 3.
- **Config-driven architecture**: `Encoder`, `Decoder`, and `Restormer`
  now take `dim`, `encoder_blocks`, `decoder_blocks`, `heads`, and
  `expansion_factor` as constructor arguments instead of hardcoding them.
  `Restormer.from_config(cfg)` builds directly from a loaded
  `configs/srrt_base.yaml` dict. Constructor defaults mirror the YAML
  defaults so existing no-arg call sites keep working.

## Phase 5/6 — Losses + training loop (2026-08-12)

### `losses/combined_loss.py`

- `L1Loss`: plain `nn.L1Loss` between prediction and target.
- `VGGPerceptualLoss`: L1 distance between frozen, ImageNet-pretrained
  VGG19 features at 4 layers (conv1_2, conv2_2, conv3_4, conv4_4 — the
  standard set for restoration/SR perceptual losses). Inputs are assumed
  in `[0,1]` (matches `data/datasets.py`'s `ToTensor()` output) and are
  re-normalized to ImageNet stats internally.
- `CombinedLoss` sums `l1_weight*L1 + perceptual_weight*L_perceptual`,
  and additionally `flow_weight*L_flow + consistency_weight*L_LCR` **only
  if** `flow_loss`/`lcr_loss` are passed to `forward()`. Since
  `rectified_flow.py`/`lcr.py` don't exist yet, `train.py` never passes
  them, so the Restormer baseline currently trains on L1 + perceptual
  only — the flow/consistency weights in the YAML are inert until Phase
  3/4 wire real values through.

### Critical finding: full 256x256 training does not fit an 8GB laptop GPU

Measured directly (`torch.cuda.max_memory_allocated`, allocator capped to
85% of VRAM to force clean OOMs instead of silent Windows shared-memory
spillover):

| Config | Peak VRAM |
|---|---|
| Restormer fwd+bwd, B=1, 256x256, no perceptual loss | 6.2 GB |
| Restormer fwd+bwd, B=2, 256x256 | OOM |
| Restormer fwd+bwd, B=1, **128x128**, no perceptual loss | 1.68 GB |
| Restormer fwd+bwd, B=2, 128x128, **+ perceptual loss** | 3.36 GB |
| Restormer fwd+bwd, B=4, 128x128, + perceptual loss | 6.43 GB |

Root cause: training (unlike inference) must retain activations for
backprop through all ~44 TransformerBlocks across the 4 resolution
levels; the highest-resolution level (dim=48 @ 256x256) dominates. This
isn't specific to our implementation — the original Restormer paper
trains with *progressive patch sizes* (128 -> 160 -> 192 -> 256 -> 320 ->
384, shrinking batch size as patch size grows) for exactly this reason;
it never trains at fixed 256x256 with a large batch either.

**Decision**: `configs/srrt_base.yaml` now sets `dataset.train.image_size:
128`, `dataset.train.batch_size: 2`, `training.grad_accum_steps: 4`
(effective batch 8, matching the paper's own stage-1 effective batch
size), leaving large VRAM headroom (~3.4GB of 8GB). Validation still runs
at 256x256 (full resolution) since it's inference-only and cheap (~500MB,
see Phase 1/2 notes). `device.max_memory_fraction: 0.85` caps the CUDA
allocator so any future misconfiguration OOMs cleanly instead of
spilling into slow/unstable shared GPU memory.

**Not implemented**: true random-crop patch training (the original
paper crops patches from full-resolution images with flip/rotation
augmentation). `data/datasets.py`'s existing `Resize`-based pipeline was
preserved as-is (per "don't modify tested code without reason") and
reused with a smaller target size instead. This is a simplification, not
the paper's exact augmentation — worth revisiting if training quality
plateaus.

### `train.py`

Verified end-to-end via `python train.py --max-steps N --max-val-batches
N` (smoke test, not real training): dataloaders build, AMP + grad
accumulation + gradient clipping (opt-in) all run, loss decreases across
steps (0.42 -> 0.36 over 2 logged epochs), checkpoints save/load/resume
correctly (`--resume checkpoints/latest.pth` correctly continued at the
next epoch with matching optimizer/scheduler state), CSV logging works,
and qualitative blur/pred/sharp triptychs save to `outputs/` with finite,
visually sane pixel values (not NaN/garbage) — the restored image already
shows partial structure recovery after just 4 optimizer steps.

No real training run (200 epochs) has been executed — that would take
substantial wall-clock time on a laptop GPU and produces the actual
PSNR/SSIM numbers this project needs; kicking it off is a separate,
explicit decision left to the user given the time/thermal cost.

### What has NOT been validated yet

- No training has been run. No PSNR/SSIM/LPIPS numbers exist. Any claim
  about restoration quality would be fabricated at this point.
- `models/seecoder.py`, `diffae.py`, `lcr.py`, `rectified_flow.py`,
  `sra.py`, `srrt.py`, `losses/combined_loss.py`, `train.py`, and
  `evaluate.py` are still empty stubs — Phase 3 onward has not started.
  Per the project's research principle, SRRT-specific modules should only
  be integrated after the Restormer baseline is trained and verified to
  actually deblur images, not just pass shape/memory tests.

---

## Phases 1-9 restructure + SRRT modules (2026-08-14)

Superseded by the fuller docs; see:
* `README.md` — status matrix, commands, measured hardware numbers, limitations
* `docs/mathematical_formulation.md` — all equations and notation
* `docs/numerical_example.md` — worked examples (all verified by tests)
* `docs/design_decisions.md` — D1..D15, every deviation and open ambiguity

Summary of this session:
* Restructured into `models/{common,backbone,modules}/` + `utils/`, `scripts/`,
  `docs/`; factored the bottleneck into its own module; behaviour preserved
  (param count unchanged at 26,126,644).
* Rewrote the dataset layer: paired random/center crops (replacing an
  aspect-distorting 256x256 resize), per-file pairing validation, augmentation,
  `{input,target,task,filename}` schema, extensible registry.
* Implemented all five SRRT modules + `models/srrt.py` integration, with
  zero-initialized gates so every ablation stage starts from the baseline.
* Added `evaluate.py` (PSNR/SSIM/LPIPS + CSV/JSON/images), ablation presets,
  runner scripts and a comparison-table generator.
* Test suite grew to 149 passing tests.

Still true: nothing has been trained; no quality claim exists.
