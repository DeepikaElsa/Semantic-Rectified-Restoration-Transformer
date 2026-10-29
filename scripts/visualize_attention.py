"""
Attention visualization for MDTA and SRA (Phase 7).

WHAT MDTA ATTENTION ACTUALLY IS -- READ THIS BEFORE PLOTTING ANYTHING
--------------------------------------------------------------------
Restormer's MDTA is TRANSPOSED (channel) attention, not spatial self-attention.
From models/common/attention.py, for input (B, C, H, W) and `num_heads` heads
with d = C / num_heads:

    q, k, v         (B, num_heads, d, H*W)
    q, k            L2-normalized along the H*W axis
    attn = softmax( q @ k^T * temperature )        (B, num_heads, d, d)
    out  = attn @ v                                (B, num_heads, d, H*W)

The H*W axis is CONTRACTED AWAY. The attention matrix is d x d over CHANNELS.
There is no HW x HW spatial attention map anywhere in this model, and the MDTA
regression test in tests/ exists specifically to fail if one ever appears.

Consequently this script NEVER reshapes an MDTA attention matrix into an image
grid or overlays it on the input. Doing so is a common and serious figure error:
it produces an authoritative-looking "attention heatmap" that is actually a
channel-covariance matrix rearranged into a square. What is plotted instead:

    * the d x d channel-attention matrix itself, per head, as a matrix
    * per-head entropy, showing how peaked or diffuse channel mixing is
    * channel importance = column mass of the attention matrix
    * the learned per-head temperature

SRA is different: its gate A_s = sigmoid(...) is a genuine per-channel,
per-position map of shape (B, C, H, W), so it IS spatially meaningful and is
shown as a spatial map (channel-averaged), correctly labelled as a gate.

    python scripts/visualize_attention.py \
        --checkpoint checkpoints/baseline_restormer/best_psnr.pth

No architectural change is required: MDTA attention is recomputed inside a hook
from the module's own q/k using the module's own weights, and SRA already
exposes its maps via `return_maps=True`.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.research import (  # noqa: E402
    apply_style, base_provenance, eval_dataset, load_checkpoint_model,
    normalize01, save_figure, select_image_ids, to_numpy_image,
)

MDTA_NOTE = ("MDTA is channel (transposed) attention: the H*W axis is contracted, "
             "so the matrix is d x d over CHANNELS, not a spatial map.")


def mdta_attention(module, x: torch.Tensor) -> torch.Tensor:
    """
    Recompute one MDTA layer's attention matrix, exactly as its forward does.

    Mirrors models/common/attention.py rather than guessing: same qkv conv, same
    depthwise conv, same reshape, same L2 normalization along H*W, same
    temperature. Returns (num_heads, d, d).

    Recomputing beats patching the module to stash `attn`, because it leaves the
    model object untouched -- important while a training run holds the same code.
    """
    b, c, h, w = x.shape
    qkv = module.qkv_dwconv(module.qkv(x))
    q, k, _ = qkv.chunk(3, dim=1)
    d = c // module.num_heads
    q = q.reshape(b, module.num_heads, d, h * w)
    k = k.reshape(b, module.num_heads, d, h * w)
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    attn = torch.matmul(q, k.transpose(-2, -1)) * module.temperature
    return attn.softmax(dim=-1)[0].detach().cpu()


def find_mdta_layers(model) -> list[tuple[str, torch.nn.Module]]:
    from models.common.attention import MDTA
    return [(n, m) for n, m in model.named_modules() if isinstance(m, MDTA)]


def entropy_per_head(attn: torch.Tensor) -> np.ndarray:
    """
    Shannon entropy of each head's attention rows, in nats, averaged over rows.

    Low entropy = a channel attends to few channels; high = diffuse mixing.
    ln(d) is the maximum, so the figure reports the ratio to that ceiling too.
    """
    p = attn.clamp_min(1e-12)
    return (-(p * p.log()).sum(dim=-1)).mean(dim=-1).numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image-id", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="test")
    ap.add_argument("--layers", type=int, default=4,
                    help="how many MDTA layers to show, evenly spaced through depth")
    ap.add_argument("--output_dir", default="research/figures/attention")
    ap.add_argument("--formats", default="png,pdf")
    args = ap.parse_args()

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg, meta = load_checkpoint_model(args.checkpoint, device)
    print(f"checkpoint {args.checkpoint}: epoch {meta['epoch']}, "
          f"modules={[k for k, v in meta['modules_enabled'].items() if v] or 'none'}")

    dataset = eval_dataset(cfg, split=args.split)
    idx = (args.image_id if args.image_id is not None
           else select_image_ids(len(dataset), 1, args.seed)[0])
    sample = dataset[idx]
    x = sample["input"].unsqueeze(0).to(device)
    filename = sample.get("filename", f"index_{idx}")

    mdta_layers = find_mdta_layers(model)
    if not mdta_layers:
        print("No MDTA layers found.")
        return 1
    print(f"{len(mdta_layers)} MDTA layers; image #{idx} ({filename})")

    # Capture each selected layer's INPUT, then recompute its attention.
    picks = np.linspace(0, len(mdta_layers) - 1, min(args.layers, len(mdta_layers)))
    picks = sorted(set(int(round(p)) for p in picks))
    selected = [mdta_layers[i] for i in picks]

    inputs: dict[str, torch.Tensor] = {}
    handles = []
    for name, mod in selected:
        # Pre-hook, not a forward hook: MDTA's attention is an internal that its
        # output does not expose, so the layer INPUT is what has to be captured.
        def pre_hook(_module, inp, _name=name):
            inputs[_name] = inp[0].detach()
        handles.append(mod.register_forward_pre_hook(pre_hook))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()

    apply_style()
    import matplotlib.pyplot as plt
    written = []

    stats = {}
    n = len(selected)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.4))
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1:
        axes = axes.reshape(2, -1)

    for j, (name, mod) in enumerate(selected):
        feat = inputs.get(name)
        if feat is None:
            continue
        attn = mdta_attention(mod, feat)          # (heads, d, d)
        heads, d, _ = attn.shape
        ent = entropy_per_head(attn)
        temps = mod.temperature.detach().cpu().flatten().numpy()
        importance = attn.sum(dim=1).mean(dim=0).numpy()   # column mass, averaged over heads

        stats[name] = {
            "shape_attention": [heads, d, d],
            "feature_shape": list(feat.shape),
            "num_heads": heads, "head_dim": d,
            "entropy_nats_per_head": ent.tolist(),
            "max_entropy_nats": float(np.log(d)),
            "temperature_per_head": temps.tolist(),
        }

        im = axes[0, j].imshow(attn[0].numpy(), cmap="cividis")
        axes[0, j].set_title(f"{name}\nhead 0: {d}x{d} CHANNEL attention", fontsize=8)
        axes[0, j].set_xlabel("key channel")
        axes[0, j].set_ylabel("query channel")
        fig.colorbar(im, ax=axes[0, j], fraction=0.046)

        axes[1, j].bar(np.arange(heads), ent, color="#0072B2")
        axes[1, j].axhline(np.log(d), color="#D55E00", linestyle="--", linewidth=1.0,
                           label=f"max = ln(d) = {np.log(d):.2f}")
        axes[1, j].set_title(f"entropy per head (d={d}, {heads} heads)", fontsize=8)
        axes[1, j].set_xlabel("head")
        axes[1, j].set_ylabel("entropy (nats)")
        axes[1, j].legend(fontsize=7)

    fig.suptitle(f"MDTA channel attention — {meta['experiment']} epoch {meta['epoch']}\n"
                 f"{MDTA_NOTE}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    prov = base_provenance(
        figure="mdta_channel_attention",
        checkpoint=args.checkpoint, experiment=meta["experiment"], epoch=meta["epoch"],
        dataset=cfg["dataset"]["name"], split=args.split,
        image_ids=[idx], filename=filename, seed=args.seed,
        crop_mode=cfg["dataset"]["val"].get("crop_mode", "center"),
        crop_size=cfg["dataset"]["val"]["crop_size"],
        normalization="input float32 [0,1]",
        attention_semantics=MDTA_NOTE,
        tensor_dimensions={
            "qkv": "(B, num_heads, head_dim, H*W)",
            "attention": "(B, num_heads, head_dim, head_dim)  <- channel x channel",
            "output": "(B, num_heads, head_dim, H*W)",
        },
        layers=stats,
        extraction="forward pre-hook captured layer input; attention recomputed from "
                   "the module's own weights. No architectural change.",
    )
    stem = f"attention_mdta_{meta['experiment']}_img{idx:04d}"
    written += save_figure(fig, args.output_dir, stem, prov, formats)
    plt.close(fig)

    # ---- SRA spatial gate, only if the checkpoint has SRA enabled -----------
    if meta["modules_enabled"].get("sra"):
        with torch.no_grad():
            out = model(x, return_features=True)
        gate = out.get("sra_attention")
        if gate is not None:
            gmap = gate[0].detach().float().cpu().abs().mean(dim=0).numpy()
            f2, a2 = plt.subplots(1, 3, figsize=(10.0, 3.4))
            a2[0].imshow(to_numpy_image(x))
            a2[0].set_title(f"Input #{idx}")
            im2 = a2[1].imshow(gmap, cmap="magma")
            a2[1].set_title(r"SRA gate $A_s$, channel-averaged"
                            f"\nshape {tuple(gate.shape)}", fontsize=9)
            f2.colorbar(im2, ax=a2[1], fraction=0.046)
            a2[2].imshow(to_numpy_image(x))
            a2[2].imshow(normalize01(gmap), cmap="magma", alpha=0.55)
            a2[2].set_title("gate overlaid on input", fontsize=9)
            for a in a2:
                a.set_xticks([])
                a.set_yticks([])
            gamma = float(model.sra.gamma.detach().cpu())
            f2.suptitle("SRA is a genuine SPATIAL gate (sigmoid, per channel and "
                        f"position), unlike MDTA.   learned $\\gamma$ = {gamma:.4g}",
                        fontsize=9)
            f2.tight_layout(rect=(0, 0, 1, 0.9))
            p2 = dict(prov)
            p2.update(figure="sra_spatial_gate",
                      gate_shape=list(gate.shape), sra_gamma=gamma,
                      attention_semantics="SRA gate is per-channel per-position "
                                          "sigmoid; spatially meaningful.")
            written += save_figure(f2, args.output_dir,
                                   f"attention_sra_{meta['experiment']}_img{idx:04d}",
                                   p2, formats)
            plt.close(f2)
    else:
        print("SRA not enabled in this checkpoint -> no spatial gate figure "
              "(expected for the pure Restormer baseline).")

    for w in written:
        print(f"wrote {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
