# Design Decisions & Documented Ambiguities

Every deliberate deviation from a published reference, and every place where the
SRRT specification was ambiguous and a choice had to be made. Nothing here is
presented as literature-backed unless a reference is named.

---

## D1. MDTA matmul order (BUG FIX — highest importance)

**Problem.** An earlier revision computed `attn = q.transpose(-2,-1) @ k`, which
contracts the channel axis and yields an `HW × HW` matrix — spatial attention.
At 256×256 that is 65,536² ≈ 4.29e9 elements (~17 GB/head fp32; ~34 GB observed
with intermediates), which aborted with a CPU allocator failure.

**Decision.** Corrected to `attn = q @ k.transpose(-2,-1)` → `(B,h,d,d)`, and
Q/K are L2-normalized along the **HW** axis (`dim=-1`), matching the published
Restormer formulation.

**Guard.** `tests/_utils.py::assert_no_quadratic_attention` monkeypatches
`torch.matmul` for the duration of a forward pass and fails if any intermediate
is square with both trailing dims above the expected channel bound. Applied in
the attention, encoder, decoder, Restormer, LCR, SRA and SRRT tests.

**Measured after fix** (256×256, dim=48, inference): MDTA 104 MB, full Restormer
494 MB, 26,126,644 params.

---

## D2. Training patch size 128 rather than the spec's 256

**Ambiguity.** Spec §6 suggests `image_size: 256, batch_size: 1`.

**Measured on the target GPU** (RTX 4070 Laptop, 8187 MiB, AMP on, allocator
capped to 85% to force clean OOMs):

| Config | Peak VRAM |
|---|---|
| Restormer fwd+bwd, B=1, 256², no perceptual | 6.2 GB |
| Restormer fwd+bwd, B=2, 256² | OOM |
| Restormer fwd+bwd, B=1, 128², no perceptual | 1.68 GB |
| Restormer fwd+bwd, B=2, 128², + perceptual | 3.36 GB |
| Restormer fwd+bwd, B=4, 128², + perceptual | 6.43 GB |

**Decision.** `crop_size: 128`, `batch_size: 2`, `grad_accum_steps: 4`
(effective batch 8). 128 is not arbitrary: it is the **first stage of the
original Restormer paper's own progressive-patch schedule**
(128→160→192→256→320→384, batch size shrinking as patch grows). The paper never
trains at a fixed 256 with a large batch either.

**Consequence.** Numbers produced at this patch size are not directly comparable
to published GoPro results. Raising `crop_size` later is a one-line config
change.

**Not implemented.** True progressive resizing across training stages (spec
mentions initial values only). Left as future work.

---

## D3. Windows shared-memory spillover (root cause of an earlier system crash)

**Observation.** A 256²/batch-2 probe reported 12.7 GB "allocated" on an 8 GB
card. On Windows/WDDM, allocations beyond dedicated VRAM silently spill into
*shared GPU memory* (system RAM) instead of raising `OutOfMemoryError`. That
spillover — not only the MDTA bug — is the likely cause of the earlier
full-system crash, since it drives host RAM pressure invisibly.

**Decision.** `device.max_memory_fraction: 0.85` calls
`torch.cuda.set_per_process_memory_fraction`, converting silent instability into
a clean, catchable OOM. `train.py` catches `torch.cuda.OutOfMemoryError` and
prints the measured table from D2 with concrete remedies.

---

## D4. Random-crop training / center-crop validation, not resize

**Problem found in the pre-existing loader.** It used
`transforms.Resize((256,256))` on 1280×720 GoPro frames, squashing 16:9 into
1:1. Aspect distortion depresses PSNR/SSIM and is not what Restormer does.

**Decision.** `data/transforms.py::PairedTransform` implements
`crop_mode ∈ {random, center, none}`:

* train — random 128² crop + paired flips and 90° rotations
* val — deterministic center crop (so val PSNR is comparable across epochs)
* `none` — full image, reflection-padded to a multiple of 8

**Critical correctness point.** Both images of a pair must receive the *same*
geometric transform. Applying torchvision transforms independently would sample
different crop windows and destroy the supervision signal.
`tests/test_dataset.py::test_random_crop_is_paired` verifies this structurally
by transforming an image against itself and requiring the outputs stay equal.

**Caveat.** Published GoPro numbers use full images. Center-crop evaluation is
*not* protocol-comparable; use `evaluate.py --full-images` (or
`val.crop_mode: none`) for that, at higher VRAM cost.

**Padding subtlety.** `F.pad(mode="reflect")` requires pad < dimension, which
fails when an image is smaller than the crop. `_pad` falls back to `replicate`
in that case (found by a test, not in production).

---

## D5. VGG19 for training, LPIPS for evaluation only

**Ambiguity.** Spec §13 names LPIPS first for the perceptual loss; spec §18 also
requires LPIPS as an evaluation metric.

**Decision.** Train with VGG19 features (`relu1_2, relu2_2, relu3_4, relu4_4`);
reserve LPIPS for evaluation. Training directly on a reported metric inflates it
and is a known methodological weakness. `loss.perceptual_type` can be switched
to `lpips` for an explicit ablation, and `tests/test_config.py` asserts the
default stays `vgg19`.

---

## D6. Bottleneck as the SRRT injection point; backbone left unmodified

**Ambiguity.** The spec's diagram places LCR/Flow *before* Restormer, but a
latent cannot literally be fed into an image-input backbone.

**Decision.** `models/srrt.py` holds a `Restormer` and calls its
encoder/bottleneck/decoder/refinement/output in sequence, splicing SRRT
components between them. The latent path meets the backbone at the
**bottleneck** (`dim*8` at `H/8`) — lowest resolution, most semantic, cheapest
place to transport a latent, and the decoder still propagates changes to full
resolution via skip connections.

The backbone is *not* edited, so `Restormer` stays a clean baseline. Verified:
`test_baseline_srrt_is_numerically_identical_to_restormer`.

---

## D7. Zero-initialized gates everywhere a module joins the network

`gamma_latent` (latent injection), `RectifiedFlow.proj_out`, and `SRA.gamma` are
all zero-initialized.

**Rationale.** An untrained module injecting random signal into a bottleneck or
decoder makes early training a recovery exercise, and it makes ablations
dishonest — enabling a module would perturb the network even if the module
learned nothing. With zero-init, every ablation stage *starts* from the baseline
and any difference is attributable to what the module learned. Asserted in
`test_sra.py::test_is_exact_noop_at_initialization` and
`test_rectified_flow.py::test_zero_init_makes_velocity_zero_and_ode_identity`.

---

## D8. LCR consistency loss: shared projection + L2 normalization + stopgrad

**Ambiguity.** Spec §9 gives `L_LCR = ||P(z_restored) - P(z_reference)||²`
without specifying `P`, or what "reference" is.

**Decisions.**
1. `P` is a **shared** 2-layer MLP on the globally pooled latent, applied to
   both arguments (a separate head per side would let the network trivially
   satisfy the loss).
2. Projections are **L2-normalized**, so the loss is `2(1 − cos θ)` and depends
   only on direction — the network cannot minimize it by shrinking latent norms
   to zero, the standard collapse mode. Verified in
   `test_lcr.py::test_consistency_loss_cannot_be_gamed_by_scaling`.
3. `z_ref` is **detached**: the clean-image latent is a fixed target, not
   something to drag toward the prediction.
4. `z_ref` = backbone latent of the clean target, available only in training.

---

## D9. During training the backbone receives z1, not the ODE solution

**Ambiguity.** What does the backbone consume while the flow is still learning?

**Decision.** With `target` supplied (training), SRRT injects the *supervised*
endpoint `z1`; at inference it injects `ODE_solve(v_theta, z0)`. This is
teacher-forcing: the backbone learns against the latent the flow is converging
toward, rather than against a poor early ODE solution.

**Known limitation (documented, unresolved).** This creates a train/inference
mismatch — the classic exposure-bias problem. Mitigations to evaluate once
training data exists: scheduled sampling between `z1` and the ODE solution, or
periodically injecting the ODE output during training. Not resolved because
choosing between them without experiments would be guesswork.

---

## D10. Module dependency rules

`LCR` requires SeeCoder **and** DiffAE (it fuses two latents); Rectified Flow
requires LCR (it transports LCR's output). Invalid combinations raise at
construction rather than failing obscurely later. This is why the ablation ladder
is A→F rather than an arbitrary subset lattice.

---

## D11. SeeCoder is deliberately not a second Restormer

A duplicated backbone would double cost and make ablation gains
uninterpretable — a gain could come from "visual context" or simply from extra
capacity. SeeCoder is a lightweight separable-conv pyramid (~335 K params vs the
backbone's 26.1 M) with one MDTA block at its deepest level.
`test_seecoder.py::test_is_not_a_restormer_copy` enforces < 10% of backbone
params.

---

## D12. DiffAE interpretation

`DiffAE` yields a **damage-aware learned representation**. Nothing in the
architecture or objective enforces disentanglement of degradation from content,
and no experiment here demonstrates it. Any stronger claim would be
unsupported. The optional decoder (`with_decoder`, default **off**) enables a
reconstruction objective; it is unvalidated and not used by default.

---

## D13. Attention is channel-wise in *every* SRRT module

LCR and SRA both use MDTA rather than spatial attention. SRA in particular runs
at full decoder resolution, where a spatial map would reproduce exactly the D1
blow-up. `test_lcr.py` and `test_sra.py` both assert this with
`assert_no_quadratic_attention`.

---

## D14. Checkpoints embed the config, so `weights_only=False`

Spec §24 requires the config inside each checkpoint, so checkpoints are not pure
tensor files. PyTorch ≥2.6 defaults `torch.load(weights_only=True)`, so loads
pass `weights_only=False` explicitly. `collect_environment` stringifies
`torch.__version__` (a `TorchVersion` object) to keep the payload plain data.

---

## D15. Not a git repository

`utils/seed.py::collect_environment` records `git_hash: null` when no repo is
present (spec §23 says "if available"). Running `git init` would make every
checkpoint traceable to a commit; recommended but not done unprompted.

---

## Open items / unvalidated choices

* Loss weights `(1.0, 0.1, 0.1, 0.1)` are the spec's initial values — untuned.
* No dataset besides GoPro is implemented (spec §15 says GoPro first). The
  registry in `data/datasets.py` is the extension point; the five planned
  datasets raise `NotImplementedError` rather than silently mis-loading.
* Heun solver implemented but untested against Euler for accuracy/cost on real
  data.
* Training throughput on this GPU is low (see README "Known limitations");
  a full 200-epoch run has not been performed, so **no quality claim exists**.
