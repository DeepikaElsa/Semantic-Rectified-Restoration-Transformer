"""
Intermediate feature-map visualization (Phase 6).

Extracts activations with forward HOOKS and with the model's existing
`return_features=True` path. It registers nothing permanent and modifies no
architecture, so it is safe to run against a checkpoint from a live experiment.

Stages covered (those that exist in the given checkpoint):

    Restormer encoder levels 1-3      backbone.encoder
    Restormer bottleneck (latent)     backbone.bottleneck
    Restormer decoder / refinement    backbone.decoder, backbone.refinement
    SeeCoder visual features          seecoder          (SRRT only)
    DiffAE damage features            diffae            (SRRT only)
    LCR fused latent                  lcr               (SRRT only)
    SRA rectified features            sra               (SRRT only)

Visualizations, per stage:

    channel-mean activation    A(x,y) = (1/C) * sum_c |F_c(x,y)|
    selected channels          the --channels highest-energy channels, raw

    python scripts/visualize_features.py \
        --checkpoint checkpoints/baseline_restormer/best_psnr.pth --image-id 98

NORMALIZATION
-------------
Each displayed map is min-max rescaled to [0,1] over that map. Feature scales
differ by orders of magnitude between stages, so a single shared scale would
render most stages black. The consequence is that BRIGHTNESS IS NOT COMPARABLE
BETWEEN PANELS -- only spatial structure within a panel is. Every figure says so
on its face. `--shared-scale` forces one scale per stage across models when
comparing two checkpoints on the same stage.

INTERPRETATION LIMIT
--------------------
A feature map that looks like an edge detector is not thereby evidence that the
network "detects edges", and a visually striking map is not evidence a module
works. These are qualitative visualizations of activation magnitude. Causal
claims require ablation, which is what the ablation ladder is for.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.research import (  # noqa: E402
    apply_style, base_provenance, channel_mean_activation, eval_dataset,
    load_checkpoint_model, normalize01, save_figure, select_image_ids, to_numpy_image,
)

DISCLAIMER = ("Qualitative activation magnitude only. Panel brightness is min-max "
              "rescaled per panel and is NOT comparable across panels.")

# submodule path -> display label. Missing paths are skipped silently, which is
# what makes one script work for the baseline and for every SRRT variant.
HOOK_TARGETS = [
    ("backbone.encoder", "Encoder"),
    ("backbone.bottleneck", "Bottleneck (latent)"),
    ("backbone.decoder", "Decoder"),
    ("backbone.refinement", "Refinement"),
    ("seecoder", "SeeCoder"),
    ("diffae", "DiffAE"),
    ("lcr", "LCR"),
    ("sra", "SRA"),
]


def get_submodule(model, path: str):
    obj = model
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def first_tensor(x):
    """
    Reduce a hook output to one (C,H,W)-ish tensor.

    Encoder returns a tuple of skips and SRA returns (tensor, dict); taking the
    first 4-D tensor gives the stage's principal feature map in every case.
    """
    if isinstance(x, torch.Tensor):
        return x if x.dim() == 4 else None
    if isinstance(x, dict):
        for v in x.values():
            t = first_tensor(v)
            if t is not None:
                return t
        return None
    if isinstance(x, (list, tuple)):
        for v in x:
            t = first_tensor(v)
            if t is not None:
                return t
    return None


def collect_features(model, x) -> dict[str, torch.Tensor]:
    """Run one forward with hooks attached, then remove every hook."""
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(label, path):
        def hook(_module, _inp, out):
            # Encoder emits (skip1, skip2, skip3, x4): record each level, since
            # the multi-scale progression is the interesting part.
            if path == "backbone.encoder" and isinstance(out, (list, tuple)):
                for i, t in enumerate(out):
                    if isinstance(t, torch.Tensor) and t.dim() == 4:
                        name = f"Encoder L{i + 1}" if i < len(out) - 1 else "Encoder out"
                        captured[name] = t.detach()
                return
            t = first_tensor(out)
            if t is not None:
                captured[label] = t.detach()
        return hook

    for path, label in HOOK_TARGETS:
        mod = get_submodule(model, path)
        if mod is None:
            continue
        handles.append(mod.register_forward_hook(make_hook(label, path)))

    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()      # never leave hooks attached to a shared model object
    return captured


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image-id", type=int, default=None,
                    help="dataset index; default = first of --seed selection")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="test")
    ap.add_argument("--channels", type=int, default=8,
                    help="how many individual channels to show for --channel-stage")
    ap.add_argument("--channel-stage", default=None,
                    help="stage label for the per-channel figure (default: bottleneck)")
    ap.add_argument("--output_dir", default="research/figures/features")
    ap.add_argument("--formats", default="png,pdf")
    args = ap.parse_args()

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg, meta = load_checkpoint_model(args.checkpoint, device)
    active = [k for k, v in meta["modules_enabled"].items() if v]
    print(f"checkpoint {args.checkpoint}: epoch {meta['epoch']}, "
          f"modules={active or 'none (pure Restormer)'}")

    dataset = eval_dataset(cfg, split=args.split)
    idx = (args.image_id if args.image_id is not None
           else select_image_ids(len(dataset), 1, args.seed)[0])
    sample = dataset[idx]
    x = sample["input"].unsqueeze(0).to(device)
    filename = sample.get("filename", f"index_{idx}")
    print(f"image #{idx} ({filename})")

    feats = collect_features(model, x)
    if not feats:
        print("No features captured -- no hook target matched this model.")
        return 1
    for name, t in feats.items():
        print(f"  {name:<22} {tuple(t.shape)}")

    apply_style()
    import matplotlib.pyplot as plt
    written = []

    # ---- channel-mean activation across every captured stage ---------------
    names = list(feats)
    ncols = min(4, len(names) + 1)
    nrows = int(np.ceil((len(names) + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    axes[0].imshow(to_numpy_image(x))
    axes[0].set_title(f"Input (degraded)\n#{idx} {filename}", fontsize=9)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    for a, name in zip(axes[1:], names):
        amap = channel_mean_activation(feats[name])
        a.imshow(normalize01(amap), cmap="magma")
        c, h, w = feats[name].shape[1:]
        a.set_title(f"{name}\nC={c}, {h}x{w}", fontsize=9)
        a.set_xticks([])
        a.set_yticks([])
    for a in axes[len(names) + 1:]:
        a.axis("off")

    fig.suptitle(f"Channel-mean activation  " r"$A(x,y)=\frac{1}{C}\sum_c |F_c(x,y)|$"
                 f"\n{meta['experiment']} epoch {meta['epoch']} — {DISCLAIMER}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    prov = base_provenance(
        figure="feature_channel_mean",
        checkpoint=args.checkpoint, experiment=meta["experiment"],
        epoch=meta["epoch"], modules_enabled=meta["modules_enabled"],
        dataset=cfg["dataset"]["name"], split=args.split,
        image_ids=[idx], filename=filename, seed=args.seed,
        crop_mode=cfg["dataset"]["val"].get("crop_mode", "center"),
        crop_size=cfg["dataset"]["val"]["crop_size"],
        normalization="input float32 [0,1]; each activation map min-max rescaled per panel",
        stages={n: list(feats[n].shape) for n in names},
        extraction="forward hooks, removed after one forward pass; no model modification",
        interpretation_limit=DISCLAIMER,
    )
    stem = f"features_{meta['experiment']}_img{idx:04d}"
    written += save_figure(fig, args.output_dir, stem, prov, formats)
    plt.close(fig)

    # ---- individual channels of one stage -----------------------------------
    stage = args.channel_stage
    if stage is None:
        stage = next((n for n in names if "Bottleneck" in n), names[-1])
    if stage not in feats:
        print(f"stage {stage!r} not captured; available: {names}")
        return 1

    t = feats[stage][0].detach().float().cpu()
    # Rank by mean absolute activation: the highest-energy channels are the ones
    # actually carrying signal, rather than an arbitrary first-k slice.
    energy = t.abs().mean(dim=(1, 2))
    order = torch.argsort(energy, descending=True)[:args.channels].tolist()

    k = len(order)
    cols = min(4, k)
    rows = int(np.ceil(k / cols))
    f2, ax2 = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.9 * rows))
    ax2 = np.atleast_1d(ax2).ravel()
    for a, ch in zip(ax2, order):
        a.imshow(normalize01(t[ch].numpy()), cmap="viridis")
        a.set_title(f"channel {ch}\nmean|F|={energy[ch]:.3f}", fontsize=8)
        a.set_xticks([])
        a.set_yticks([])
    for a in ax2[k:]:
        a.axis("off")
    f2.suptitle(f"{stage}: {k} highest-energy channels of {t.shape[0]}"
                f"\n{meta['experiment']} epoch {meta['epoch']} — {DISCLAIMER}", fontsize=10)
    f2.tight_layout(rect=(0, 0, 1, 0.9))

    prov2 = dict(prov)
    prov2.update(figure="feature_channels", stage=stage,
                 channels_shown=order, channel_ranking="mean |activation|, descending")
    written += save_figure(f2, args.output_dir,
                           f"{stem}_{stage.split()[0].lower()}_channels", prov2, formats)
    plt.close(f2)

    for w in written:
        print(f"wrote {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
