"""
Qualitative restoration comparison grids (Phases 3 and 8).

Produces, for a fixed set of deterministically chosen GoPro test images:

    row 1:  Degraded | Ground truth | Restored (per model)
    row 2:  Error map | Zoomed GT   | Zoomed prediction (per model)

and a combined multi-image grid. Works with one model or many; with several,
every model is shown on THE SAME images and THE SAME zoom window.

    # baseline only
    python scripts/generate_qualitative_grid.py \
        --model Restormer=checkpoints/baseline_restormer/best_psnr.pth \
        --num_samples 6

    # once SRRT checkpoints exist, side by side
    python scripts/generate_qualitative_grid.py \
        --model Restormer=checkpoints/baseline_restormer/best_psnr.pth \
        --model SRRT=checkpoints/full_srrt/best_psnr.pth \
        --num_samples 6 --seed 42

FAIRNESS RULES ENFORCED IN CODE, not left to the operator
---------------------------------------------------------
* Image IDs come from `select_image_ids(len(dataset), n, seed)`, which sees no
  model and no metric. Passing more models cannot change which images appear.
* The zoom window is chosen from the GROUND TRUTH gradient energy alone. Picking
  it from a model's output would let one model choose the region it wins on.
* Every model reads the same dataset object, hence identical crop, resize and
  normalization.
* No contrast stretching is applied to any image panel. Error maps share one
  colour scale across models (`--error-vmax`, or the max over all models shown),
  so a darker error map means less error rather than a different scale.
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


def parse_models(specs: list[str]) -> list[tuple[str, str]]:
    """`--model Name=path` -> [(Name, path)], preserving command-line order."""
    out = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--model must be NAME=path/to/checkpoint, got {spec!r}")
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            raise SystemExit(f"checkpoint not found: {path}")
        out.append((name.strip(), path.strip()))
    return out


def gradient_energy(gray: np.ndarray) -> np.ndarray:
    """Squared finite-difference gradient magnitude; the 'detail' surface."""
    gy, gx = np.gradient(gray.astype(np.float64))
    return gx ** 2 + gy ** 2


def pick_zoom_window(gt_img: np.ndarray, size: int) -> tuple[int, int, int]:
    """
    Locate the most detailed square window in the GROUND TRUTH.

    Detail = summed gradient energy, which peaks on exactly the structures the
    paper needs to show: edges, text, fine texture, motion boundaries and
    repetitive patterns. Computed via a summed-area table so the search is exact
    (every candidate window) rather than a heuristic over a few guesses.

    Depends only on the ground truth, so it is identical for every model and
    reproducible from the image ID alone. Returns (top, left, size).
    """
    h, w, _ = gt_img.shape
    size = min(size, h, w)
    lum = 0.299 * gt_img[..., 0] + 0.587 * gt_img[..., 1] + 0.114 * gt_img[..., 2]
    energy = gradient_energy(lum)

    integral = np.pad(energy, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    # Window sums for every top-left position, via inclusion-exclusion.
    sums = (integral[size:, size:] - integral[:-size, size:]
            - integral[size:, :-size] + integral[:-size, :-size])
    top, left = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return int(top), int(left), size


def worst_case_ids(dataset, summary_path: str, n: int) -> list[int]:
    """
    Dataset indices of the worst-PSNR images recorded in an evaluate.py summary.

    A failure-case figure is the one place where non-random selection is correct,
    because the claim being made is "these are the hardest cases", not "these are
    typical". The figure and its provenance both say so explicitly, and the
    selection comes from ONE model's summary applied to every model shown -- so
    it is still the same images for all of them, never per-model cherry-picking.
    """
    import json
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    names = summary.get("worst_5_by_psnr") or []
    if not names:
        raise SystemExit(f"{summary_path} has no 'worst_5_by_psnr' list.")

    index_of = {os.path.basename(inp): i for i, (inp, _tgt) in enumerate(dataset.pairs)}
    ids = []
    for name in names[:n]:
        key = os.path.basename(str(name))
        if key in index_of:
            ids.append(index_of[key])
        else:
            print(f"  warning: {key!r} from the summary is not in this split; skipped")
    if not ids:
        raise SystemExit("None of the worst-case filenames matched this dataset split.")
    return sorted(ids)


def draw_box(ax, top, left, size, color="#D55E00"):
    import matplotlib.patches as patches
    ax.add_patch(patches.Rectangle((left, top), size, size, linewidth=1.4,
                                   edgecolor=color, facecolor="none"))


def show(ax, img, title):
    ax.imshow(img, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", action="append", required=True,
                    help="NAME=path/to/checkpoint (repeatable)")
    ap.add_argument("--num_samples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="test")
    ap.add_argument("--output_dir", default="research/qualitative")
    ap.add_argument("--image-ids", default=None,
                    help="comma-separated dataset indices; overrides --seed selection "
                         "(use to reproduce an exact earlier figure)")
    ap.add_argument("--failure-cases", default=None, metavar="SUMMARY_JSON",
                    help="show the WORST images by PSNR listed in an evaluate.py "
                         "summary.json instead of a random sample (failure-case figure)")
    ap.add_argument("--zoom", type=int, default=64, help="zoom window size in pixels")
    ap.add_argument("--error-mode", default="abs_mean",
                    choices=["abs_mean", "squared_mean", "normalized"])
    ap.add_argument("--error-vmax", type=float, default=None,
                    help="fix the error colour scale (default: max across shown models)")
    ap.add_argument("--full-images", action="store_true",
                    help="use uncropped images instead of the configured val crop")
    ap.add_argument("--formats", default="png,pdf")
    ap.add_argument("--stem", default="qualitative_grid",
                    help="filename stem for the combined contact sheet")
    ap.add_argument("--with-error", action="store_true",
                    help="append one error-map column per model to the contact "
                         "sheet, on a single shared colour scale")
    args = ap.parse_args()

    models = parse_models(args.model)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded = []
    for name, path in models:
        model, cfg, meta = load_checkpoint_model(path, device)
        loaded.append((name, model, meta))
        print(f"loaded {name}: epoch {meta['epoch']}, "
              f"modules={[k for k, v in meta['modules_enabled'].items() if v] or 'none'}")

    # All models share the dataset built from the FIRST checkpoint's config, so a
    # differently-configured second checkpoint cannot change the preprocessing.
    _, cfg0, _ = load_checkpoint_model(models[0][1], device)
    crop_mode = "none" if args.full_images else None
    dataset = eval_dataset(cfg0, split=args.split, crop_mode=crop_mode)

    if args.image_ids:
        ids = [int(x) for x in args.image_ids.split(",") if x.strip()]
        selection = "explicit --image-ids"
    elif args.failure_cases:
        ids = worst_case_ids(dataset, args.failure_cases, args.num_samples)
        selection = f"worst PSNR from {args.failure_cases} (FAILURE CASES, not a random sample)"
    else:
        ids = select_image_ids(len(dataset), args.num_samples, args.seed)
        selection = f"random, seed={args.seed}"
    print(f"dataset {args.split}: {len(dataset)} images; using ids {ids}  [{selection}]")

    apply_style()
    import matplotlib.pyplot as plt

    os.makedirs(args.output_dir, exist_ok=True)
    written = []
    per_image_records = []

    for idx in ids:
        sample = dataset[idx]
        degraded = sample["input"].unsqueeze(0).to(device)
        gt = sample["target"].unsqueeze(0).to(device)
        filename = sample.get("filename", f"index_{idx}")

        gt_img = to_numpy_image(gt)
        deg_img = to_numpy_image(degraded)
        top, left, zsize = pick_zoom_window(gt_img, args.zoom)

        preds, errors = {}, {}
        for name, model, _ in loaded:
            with torch.no_grad():
                out = model(degraded)
                pred = out["output"] if isinstance(out, dict) else out
            preds[name] = to_numpy_image(pred)
            errors[name] = error_map(pred, gt, args.error_mode)

        # One shared colour scale so error maps are comparable BETWEEN models.
        vmax = args.error_vmax or max(float(e.max()) for e in errors.values())

        ncols = 2 + len(loaded)
        fig, axes = plt.subplots(2, ncols, figsize=(2.6 * ncols, 6.2))
        if ncols == 1:
            axes = axes.reshape(2, 1)

        show(axes[0, 0], deg_img, "Degraded (blurred)")
        draw_box(axes[0, 0], top, left, zsize)
        show(axes[0, 1], gt_img, "Ground truth")
        draw_box(axes[0, 1], top, left, zsize)
        for j, (name, _, _) in enumerate(loaded):
            show(axes[0, 2 + j], preds[name], f"{name} (restored)")
            draw_box(axes[0, 2 + j], top, left, zsize)

        zoom = lambda a: a[top:top + zsize, left:left + zsize]
        show(axes[1, 0], zoom(deg_img), f"Zoom: degraded ({zsize}px)")
        show(axes[1, 1], zoom(gt_img), "Zoom: ground truth")
        for j, (name, _, _) in enumerate(loaded):
            show(axes[1, 2 + j], zoom(preds[name]), f"Zoom: {name}")

        fig.suptitle(f"{filename}  (dataset index {idx})", fontsize=10)
        # h_pad keeps the row-2 titles off the row-1 images: with ticks disabled
        # matplotlib underestimates how much room each panel's title needs.
        fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.2)

        prov = base_provenance(
            figure="qualitative_comparison",
            dataset=cfg0["dataset"]["name"], split=args.split,
            image_index=idx, filename=filename,
            image_ids=ids, seed=args.seed, selection_method=selection,
            resolution=list(gt_img.shape[:2]),
            crop_mode="none (full image)" if args.full_images
                      else cfg0["dataset"]["val"].get("crop_mode", "center"),
            normalization="float32 in [0,1]; predictions clipped to [0,1] for display",
            zoom_window={"top": top, "left": left, "size": zsize,
                         "selected_by": "max gradient energy of the GROUND TRUTH "
                                        "(model-independent, exact search)"},
            error_mode=args.error_mode, error_vmax=vmax,
            models=[{"name": n, **m} for n, _, m in loaded],
        )
        written += save_figure(fig, args.output_dir, f"sample_{idx:04d}", prov, formats)
        plt.close(fig)

        per_image_records.append({"index": idx, "filename": filename,
                                  "zoom": [top, left, zsize]})

        # Error-map panel per image, one column per model, shared scale.
        fe, ae = plt.subplots(1, len(loaded) + 1, figsize=(2.7 * (len(loaded) + 1), 3.0))
        ae = np.atleast_1d(ae)
        show(ae[0], gt_img, "Ground truth")
        for j, (name, _, _) in enumerate(loaded):
            im = ae[j + 1].imshow(errors[name], cmap="inferno", vmin=0.0, vmax=vmax)
            ae[j + 1].set_title(f"{name}: |error|\nMAE={errors[name].mean():.4f}", fontsize=9)
            ae[j + 1].set_xticks([])
            ae[j + 1].set_yticks([])
        fe.colorbar(im, ax=ae.tolist(), fraction=0.025, pad=0.01,
                    label=f"{args.error_mode} error (shared scale)")
        fe.suptitle(f"{filename} — pixel-wise error", fontsize=10)
        pe = dict(prov)
        pe["figure"] = "qualitative_error_maps"
        written += save_figure(fe, args.output_dir, f"sample_{idx:04d}_error", pe, formats)
        plt.close(fe)

    # Combined contact sheet: one row per image, columns = degraded/GT/models,
    # optionally followed by one error column per model. Error columns share a
    # single colour scale computed across ALL models and ALL rows, so a darker
    # panel means less error and never a rescaled axis.
    n_err = len(loaded) if args.with_error else 0
    ncols = 2 + len(loaded) + n_err
    fg, axg = plt.subplots(len(ids), ncols, figsize=(2.5 * ncols, 2.5 * len(ids)))
    axg = np.atleast_2d(axg)

    # Pass 1: compute every panel up front so the shared error scale is known
    # before anything is drawn.
    sheet: list[dict] = []
    for idx in ids:
        sample = dataset[idx]
        degraded = sample["input"].unsqueeze(0).to(device)
        gt = sample["target"].unsqueeze(0).to(device)
        row = {"idx": idx, "deg": to_numpy_image(degraded),
               "gt": to_numpy_image(gt), "pred": {}, "err": {}}
        for name, model, _ in loaded:
            with torch.no_grad():
                out = model(degraded)
                pred = out["output"] if isinstance(out, dict) else out
            row["pred"][name] = to_numpy_image(pred)
            if args.with_error:
                row["err"][name] = error_map(pred, gt, args.error_mode)
        sheet.append(row)

    sheet_vmax = args.error_vmax
    if args.with_error and sheet_vmax is None:
        sheet_vmax = max(float(e.max()) for r in sheet for e in r["err"].values())

    im_err = None
    for i, row in enumerate(sheet):
        show(axg[i, 0], row["deg"], "Degraded" if i == 0 else "")
        show(axg[i, 1], row["gt"], "Ground truth" if i == 0 else "")
        for j, (name, _model, _) in enumerate(loaded):
            show(axg[i, 2 + j], row["pred"][name], name if i == 0 else "")
        if args.with_error:
            for j, (name, _model, _) in enumerate(loaded):
                ax = axg[i, 2 + len(loaded) + j]
                im_err = ax.imshow(row["err"][name], cmap="inferno",
                                   vmin=0.0, vmax=sheet_vmax)
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"{name}: |error|", fontsize=9)
        axg[i, 0].set_ylabel(f"#{row['idx']}", fontsize=8)

    fg.suptitle("Qualitative restoration comparison — identical images for all models",
                fontsize=11)
    fg.tight_layout(rect=(0, 0, 1, 0.97))
    if im_err is not None:
        fg.colorbar(im_err, ax=axg.ravel().tolist(), fraction=0.012, pad=0.01,
                    label=f"{args.error_mode} error (one shared scale)")
    prov_all = base_provenance(
        figure="qualitative_grid_combined",
        dataset=cfg0["dataset"]["name"], split=args.split,
        images=per_image_records, seed=args.seed, selection_method=selection,
        models=[{"name": n, **m} for n, _, m in loaded],
        note="Image IDs chosen before any model was loaded; identical for every model shown.",
    )
    prov_all["with_error_columns"] = bool(args.with_error)
    if args.with_error:
        prov_all["error_mode"] = args.error_mode
        prov_all["error_vmax_shared"] = sheet_vmax
    written += save_figure(fg, args.output_dir, args.stem, prov_all, formats)
    plt.close(fg)

    for w in written:
        print(f"wrote {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
