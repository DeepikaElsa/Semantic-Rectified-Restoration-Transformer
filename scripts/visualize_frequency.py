"""
Frequency-domain analysis of restoration (Phase 5).

Motion blur is approximately a low-pass convolution, so the question "did the
model put the high frequencies back?" is directly inspectable. For a luminance
image I this script displays

    F(u,v) = FFT2(I),      shown as   log(1 + |F(u,v)|),   zero frequency centred

for the degraded input, the ground truth, and each model's output, plus an
azimuthally averaged radial profile that collapses each 2-D spectrum into a
curve so the comparison is readable rather than three similar grey squares.

    python scripts/visualize_frequency.py \
        --model Restormer=checkpoints/baseline_restormer/best_psnr.pth \
        --num_samples 4

INTERPRETATION LIMIT -- READ BEFORE CITING THIS FIGURE
------------------------------------------------------
Recovering high-frequency energy is NOT evidence of better restoration, and this
script deliberately makes no such claim. A model can raise high-frequency energy
by hallucinating texture, by ringing, or by amplifying noise, all of which move
the spectrum toward the ground truth's while making the image worse. The spectrum
also discards all phase information, and phase is where structure lives.

Treat this as descriptive analysis of what the model does to the frequency
content. Claims about restoration quality must rest on PSNR/SSIM/LPIPS, and
claims about perceptual quality on LPIPS or a human study.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.research import (  # noqa: E402
    MODEL_COLORS, OKABE_ITO, apply_style, base_provenance, eval_dataset,
    fft_log_magnitude, load_checkpoint_model, radial_profile, save_figure,
    select_image_ids, to_numpy_image,
)
from scripts.generate_qualitative_grid import parse_models, show  # noqa: E402

DISCLAIMER = ("Descriptive only: high-frequency energy alone does not imply better "
              "restoration (hallucination and ringing also raise it).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", action="append", required=True,
                    help="NAME=path/to/checkpoint (repeatable)")
    ap.add_argument("--num_samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="test")
    ap.add_argument("--output_dir", default="research/figures")
    ap.add_argument("--image-ids", default=None)
    ap.add_argument("--formats", default="png,pdf")
    args = ap.parse_args()

    models = parse_models(args.model)
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

    apply_style()
    import matplotlib.pyplot as plt
    written = []

    spectra = {}
    for idx in ids:
        s = dataset[idx]
        deg = s["input"].unsqueeze(0).to(device)
        gt = s["target"].unsqueeze(0).to(device)
        entry = {"filename": s.get("filename", f"index_{idx}"),
                 "images": {"Degraded": to_numpy_image(deg),
                            "Ground truth": to_numpy_image(gt)}}
        for name, model, _ in loaded:
            with torch.no_grad():
                out = model(deg)
                pred = out["output"] if isinstance(out, dict) else out
            entry["images"][name] = to_numpy_image(pred)
        entry["spectra"] = {k: fft_log_magnitude(v) for k, v in entry["images"].items()}
        spectra[idx] = entry

    panel_names = ["Degraded", "Ground truth"] + [n for n, _, _ in loaded]

    # ---- per-image spectra grid -------------------------------------------
    ncols = len(panel_names)
    fig, axes = plt.subplots(len(ids), ncols, figsize=(2.7 * ncols, 2.9 * len(ids)))
    axes = np.atleast_2d(axes)
    for i, idx in enumerate(ids):
        e = spectra[idx]
        # One scale per image so the panels within a row are comparable.
        vmax = max(float(sp.max()) for sp in e["spectra"].values())
        for j, name in enumerate(panel_names):
            im = axes[i, j].imshow(e["spectra"][name], cmap="viridis", vmin=0, vmax=vmax)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(name, fontsize=9)
        axes[i, 0].set_ylabel(f"#{idx}", fontsize=8)
        fig.colorbar(im, ax=list(axes[i, :]), fraction=0.02, pad=0.01)
    fig.suptitle(r"Fourier magnitude  $\log(1+|F(u,v)|)$, zero frequency centred"
                 f"\n{DISCLAIMER}", fontsize=10)

    prov = base_provenance(
        figure="frequency_spectra",
        dataset=cfg0["dataset"]["name"], split=args.split,
        image_ids=ids, seed=args.seed,
        filenames=[spectra[i]["filename"] for i in ids],
        crop_mode=cfg0["dataset"]["val"].get("crop_mode", "center"),
        crop_size=cfg0["dataset"]["val"]["crop_size"],
        normalization="float32 [0,1]; Rec.601 luminance; log(1+|FFT2|), fftshift",
        colour_scale="one scale per image row",
        interpretation_limit=DISCLAIMER,
        models=[{"name": n, **m} for n, _, m in loaded],
    )
    written += save_figure(fig, args.output_dir, "frequency_spectra", prov, formats)
    plt.close(fig)

    # ---- radial profiles ---------------------------------------------------
    # Averaged over the selected images: a single image's profile is noisy and
    # invites over-reading a bump that is not reproducible.
    profiles = {name: [] for name in panel_names}
    for idx in ids:
        for name in panel_names:
            r, p = radial_profile(spectra[idx]["spectra"][name])
            profiles[name].append(p)

    nbins = min(len(p) for ps in profiles.values() for p in ps)
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.0))

    colors = {"Degraded": OKABE_ITO["orange"], "Ground truth": OKABE_ITO["black"]}
    for n, _, _ in loaded:
        colors[n] = MODEL_COLORS.get(n.lower().replace(" ", "_"), OKABE_ITO["blue"])

    mean_prof = {}
    for name in panel_names:
        arr = np.vstack([p[:nbins] for p in profiles[name]])
        mp = np.nanmean(arr, axis=0)
        mean_prof[name] = mp
        style = "--" if name == "Ground truth" else "-"
        a1.plot(np.arange(nbins), mp, style, color=colors[name], label=name)
    a1.set_xlabel("Radial spatial frequency (pixels from DC)")
    a1.set_ylabel(r"mean $\log(1+|F|)$")
    a1.set_title(f"Azimuthally averaged spectrum (mean of {len(ids)} images)")
    a1.legend()

    # Difference from ground truth makes the gap legible; zero is the target.
    for name in panel_names:
        if name == "Ground truth":
            continue
        a2.plot(np.arange(nbins), mean_prof[name] - mean_prof["Ground truth"],
                color=colors[name], label=name)
    a2.axhline(0.0, color="0.3", linestyle="--", linewidth=0.9)
    a2.set_xlabel("Radial spatial frequency (pixels from DC)")
    a2.set_ylabel(r"$\Delta \log(1+|F|)$ vs ground truth")
    a2.set_title("Deviation from ground-truth spectrum (0 = matches GT)")
    a2.legend()
    fig2.suptitle(DISCLAIMER, fontsize=9)
    fig2.tight_layout(rect=(0, 0, 1, 0.93))

    prov2 = dict(prov)
    prov2["figure"] = "frequency_radial_profile"
    prov2["aggregation"] = f"mean over {len(ids)} images, azimuthal average per image"
    written += save_figure(fig2, args.output_dir, "frequency_radial_profile", prov2, formats)
    plt.close(fig2)

    for w in written:
        print(f"wrote {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
