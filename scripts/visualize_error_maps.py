"""
Pixel-wise restoration error maps (Phase 4).

Shows WHERE a model's restoration is wrong, which is the evidence needed to argue
that a module helps on specific structures rather than on average.

Error definitions (the figure and its provenance always state which is used):

    absolute     E(x,y) = (1/3) * sum_c |I_pred_c(x,y) - I_gt_c(x,y)|
    squared      E(x,y) = (1/3) * sum_c ( I_pred_c(x,y) - I_gt_c(x,y) )^2
    normalized   absolute, rescaled by its own per-image maximum

All are computed on float images in [0,1] AFTER clipping predictions to [0,1],
so they agree with the range the PSNR/SSIM/LPIPS metrics see.

    python scripts/visualize_error_maps.py \
        --model Restormer=checkpoints/baseline_restormer/best_psnr.pth \
        --num_samples 4

COLOUR-SCALE POLICY
-------------------
`absolute` and `squared` maps are drawn on ONE shared scale across all models and
(with --shared-scale-across-images) all images, so a visually darker map really
does mean smaller error. `normalized` is rescaled per image for visibility and is
therefore NOT comparable between panels; the script labels it as such on the
figure itself so a reader cannot mistake it for a like-for-like comparison.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.research import (  # noqa: E402
    apply_style, base_provenance, error_map, eval_dataset, load_checkpoint_model,
    save_figure, select_image_ids, to_numpy_image,
)
from scripts.generate_qualitative_grid import parse_models, show  # noqa: E402

MODE_LABEL = {
    "abs_mean": r"$E=\frac{1}{3}\sum_c |I_{pred,c}-I_{gt,c}|$",
    "squared_mean": r"$E=\frac{1}{3}\sum_c (I_{pred,c}-I_{gt,c})^2$",
    "normalized": r"$E=|\cdot|$ rescaled per image (NOT comparable across panels)",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", action="append", required=True,
                    help="NAME=path/to/checkpoint (repeatable)")
    ap.add_argument("--num_samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="test")
    ap.add_argument("--output_dir", default="research/figures")
    ap.add_argument("--image-ids", default=None)
    ap.add_argument("--modes", default="abs_mean,squared_mean",
                    help="comma-separated subset of abs_mean,squared_mean,normalized")
    ap.add_argument("--shared-scale-across-images", action="store_true",
                    help="one colour scale for the whole figure instead of per image")
    ap.add_argument("--formats", default="png,pdf")
    args = ap.parse_args()

    models = parse_models(args.model)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded = []
    for name, path in models:
        model, cfg, meta = load_checkpoint_model(path, device)
        loaded.append((name, model, meta))
    _, cfg0, _ = load_checkpoint_model(models[0][1], device)

    dataset = eval_dataset(cfg0, split=args.split)
    ids = ([int(x) for x in args.image_ids.split(",") if x.strip()] if args.image_ids
           else select_image_ids(len(dataset), args.num_samples, args.seed))
    print(f"dataset {args.split}: {len(dataset)} images; using ids {ids}")

    # Precompute everything first: a shared colour scale needs the global max
    # before the first panel is drawn.
    data = {}
    for idx in ids:
        s = dataset[idx]
        deg = s["input"].unsqueeze(0).to(device)
        gt = s["target"].unsqueeze(0).to(device)
        entry = {"gt": to_numpy_image(gt), "deg": to_numpy_image(deg),
                 "filename": s.get("filename", f"index_{idx}"), "models": {}}
        for name, model, _ in loaded:
            with torch.no_grad():
                out = model(deg)
                pred = out["output"] if isinstance(out, dict) else out
            entry["models"][name] = {
                "pred": to_numpy_image(pred),
                **{m: error_map(pred, gt, m) for m in modes},
            }
        data[idx] = entry

    global_vmax = {m: max(float(data[i]["models"][n][m].max())
                          for i in ids for n, _, _ in loaded) for m in modes}

    apply_style()
    import matplotlib.pyplot as plt
    written = []

    for mode in modes:
        ncols = 2 + len(loaded)
        fig, axes = plt.subplots(len(ids), ncols,
                                 figsize=(2.7 * ncols, 2.9 * len(ids)))
        axes = np.atleast_2d(axes)

        for i, idx in enumerate(ids):
            e = data[idx]
            vmax = (global_vmax[mode] if args.shared_scale_across_images
                    else max(float(e["models"][n][mode].max()) for n, _, _ in loaded))
            if mode == "normalized":
                vmax = 1.0

            show(axes[i, 0], e["gt"], "Ground truth" if i == 0 else "")
            show(axes[i, 1], e["models"][loaded[0][0]]["pred"],
                 f"{loaded[0][0]} prediction" if i == 0 else "")
            for j, (name, _, _) in enumerate(loaded):
                em = e["models"][name][mode]
                im = axes[i, 2 + j].imshow(em, cmap="inferno", vmin=0.0, vmax=vmax)
                axes[i, 2 + j].set_xticks([])
                axes[i, 2 + j].set_yticks([])
                axes[i, 2 + j].set_title(
                    (f"{name}: error\n" if i == 0 else "") + f"mean={em.mean():.4f}",
                    fontsize=8)
            axes[i, 0].set_ylabel(f"#{idx}", fontsize=8)

            # A per-row colorbar unless the scale is genuinely global. One shared
            # bar over rows that each used their own vmax would misstate the
            # values a reader reads off the colours.
            if not args.shared_scale_across_images or mode == "normalized":
                fig.colorbar(im, ax=list(axes[i, 2:]), fraction=0.03, pad=0.01)

        if args.shared_scale_across_images and mode != "normalized":
            fig.colorbar(im, ax=list(axes.ravel()), fraction=0.02, pad=0.01,
                         label="error (shared across all panels)")
        scale_note = ("one scale for the whole figure"
                      if args.shared_scale_across_images and mode != "normalized"
                      else "one scale per image row" if mode != "normalized"
                      else "per-image rescaling")
        fig.suptitle(f"Restoration error — {mode}\n{MODE_LABEL[mode]}   ({scale_note})",
                     fontsize=10)

        prov = base_provenance(
            figure=f"error_maps_{mode}",
            dataset=cfg0["dataset"]["name"], split=args.split,
            image_ids=ids, seed=args.seed,
            filenames=[data[i]["filename"] for i in ids],
            crop_mode=cfg0["dataset"]["val"].get("crop_mode", "center"),
            crop_size=cfg0["dataset"]["val"]["crop_size"],
            normalization="float32 in [0,1]; predictions clipped to [0,1]",
            error_definition=MODE_LABEL[mode],
            colour_scale=scale_note,
            vmax=global_vmax[mode] if args.shared_scale_across_images else "per-row",
            mean_error_per_model={
                n: float(np.mean([data[i]["models"][n][mode].mean() for i in ids]))
                for n, _, _ in loaded},
            models=[{"name": n, **m} for n, _, m in loaded],
        )
        written += save_figure(fig, args.output_dir, f"error_maps_{mode}", prov, formats)
        plt.close(fig)

    for w in written:
        print(f"wrote {w}")
    print("\nMean error over the shown images (lower is better):")
    for mode in modes:
        for name, _, _ in loaded:
            v = np.mean([data[i]["models"][name][mode].mean() for i in ids])
            print(f"  {mode:<14} {name:<12} {v:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
