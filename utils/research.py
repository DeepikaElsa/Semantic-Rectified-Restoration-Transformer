"""
Shared plumbing for the publication-figure pipeline.

Every research script imports from here so that figures across the paper share
one typography, one colour convention, one error-map definition, and one
deterministic sample-selection rule. Duplicating any of those per-script is how
two figures end up silently comparing different images or different crops.

Three invariants this module exists to enforce:

1. **Read-only access to live experiment logs.** The 60-epoch baseline appends to
   `logs/<exp>/train_log.csv` while these scripts run. `read_train_log` snapshots
   the bytes and discards a torn trailing line rather than opening the file for
   anything but reading.

2. **Deterministic, model-independent sample selection.** `select_image_ids`
   depends only on (seed, dataset length, count) -- never on a model, a metric,
   or a checkpoint. The same IDs therefore come out for Restormer and for every
   SRRT variant, which is what makes a qualitative comparison honest.

3. **Provenance on every figure.** `save_figure` refuses to write a figure
   without a provenance dict and emits it as a JSON sidecar next to the image.
"""

from __future__ import annotations

import csv
import io
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Typography and colour
# --------------------------------------------------------------------------

# One serif-free, colour-blind-safe scheme reused everywhere. Okabe-Ito, which
# stays distinguishable in deuteranopia and in greyscale print.
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}

# Fixed per-metric colours so PSNR is the same blue in every figure of the paper.
METRIC_COLORS = {
    "train_loss": OKABE_ITO["blue"],
    "val_loss": OKABE_ITO["vermillion"],
    "val_psnr": OKABE_ITO["green"],
    "val_ssim": OKABE_ITO["purple"],
    "val_lpips": OKABE_ITO["orange"],
    "lr": OKABE_ITO["black"],
}

# Fixed per-model colours so "Restormer" is the same colour in every ablation plot.
MODEL_COLORS = {
    "baseline_restormer": OKABE_ITO["blue"],
    "seecoder": OKABE_ITO["sky"],
    "seecoder_diffae": OKABE_ITO["green"],
    "seecoder_diffae_lcr": OKABE_ITO["yellow"],
    "seecoder_diffae_lcr_flow": OKABE_ITO["orange"],
    "full_srrt": OKABE_ITO["vermillion"],
}

# Human-readable names for the ablation ladder, in the order the paper reports it.
MODEL_LABELS = {
    "baseline_restormer": "Restormer",
    "seecoder": "+ SeeCoder",
    "seecoder_diffae": "+ DiffAE",
    "seecoder_diffae_lcr": "+ LCR",
    "seecoder_diffae_lcr_flow": "+ Rectified Flow",
    "full_srrt": "SRRT (+ SRA)",
}

# Which direction is better. Used to decide arrows in tables and to avoid the
# classic bug of bolding the *worst* LPIPS.
METRIC_DIRECTION = {"psnr": "higher", "ssim": "higher", "lpips": "lower"}

DPI_RASTER = 300


def apply_style() -> None:
    """Install the paper rcParams. Call once, before creating any figure."""
    import matplotlib
    matplotlib.use("Agg")           # no display on this machine; must precede pyplot
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": DPI_RASTER,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "font.family": "DejaVu Sans",   # ships with matplotlib; no missing-glyph boxes
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.6,
        "lines.markersize": 3.5,
        "figure.autolayout": False,
        # Keep text as text in the PDF so the publisher can search/edit it.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

def git_hash() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def base_provenance(**fields: Any) -> dict[str, Any]:
    """
    Start a provenance record with the fields every figure should carry.

    Callers add figure-specific keys: checkpoint, dataset, image_ids, seed,
    resolution, normalization, crop_mode.
    """
    record = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": os.path.basename(sys.argv[0]) if sys.argv else "python",
        "argv": sys.argv[1:],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_hash": git_hash(),
    }
    record.update(fields)
    return record


def save_figure(fig, out_dir: str, stem: str, provenance: dict[str, Any],
                formats: Sequence[str] = ("png", "pdf")) -> list[str]:
    """
    Write a figure in each requested format plus a `<stem>.json` provenance sidecar.

    Provenance is mandatory: a figure whose origin cannot be reconstructed is not
    usable in a paper, and making it optional guarantees it gets skipped.
    """
    if not provenance:
        raise ValueError(f"refusing to save {stem}: provenance is required")

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for ext in formats:
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path)
        written.append(path)

    meta_path = os.path.join(out_dir, f"{stem}.json")
    with io.open(meta_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, default=str)
    written.append(meta_path)
    return written


# --------------------------------------------------------------------------
# Reading experiment logs (never writing them)
# --------------------------------------------------------------------------

def to_float(v: Any) -> float | None:
    """Parse a CSV cell, rejecting NaN so a NaN never silently plots as a gap."""
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def resolve_train_log(experiment: str) -> str:
    """
    Accept an experiment NAME, a results directory, or a direct CSV path.

    Supports `--experiment results/restormer_baseline` (as the paper pipeline
    spells it) and `--experiment baseline_restormer` (as train.py spells it).
    """
    cands = []
    if experiment.endswith(".csv"):
        cands.append(experiment)
    else:
        cands += [
            os.path.join(experiment, "metrics", "train_log.csv"),
            os.path.join(experiment, "train_log.csv"),
            os.path.join("logs", experiment, "train_log.csv"),
            os.path.join("results", experiment, "metrics", "train_log.csv"),
        ]
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"No train_log.csv for {experiment!r}. Tried:\n  " + "\n  ".join(cands))


def read_train_log(path: str) -> list[dict[str, Any]]:
    """
    Read a training CSV that a live run may be appending to, without disturbing it.

    Opened read-only and snapshotted in one go. A trailing line that is short on
    fields is a torn write caught mid-append and is dropped -- plotting it would
    show a phantom epoch with null metrics.

    Later rows win on duplicate epochs: a resumed run re-runs no epoch, but a
    smoke test writing into the same log would, and the newest row is the real one.
    """
    with io.open(path, "r", encoding="utf-8", newline="") as f:
        raw = f.read()

    reader = csv.DictReader(io.StringIO(raw))
    rows = list(reader)
    if not rows:
        return []

    n_fields = len(reader.fieldnames or [])
    last = rows[-1]
    if last.get("epoch") in (None, "") or len([v for v in last.values() if v is not None]) < n_fields:
        rows = rows[:-1]

    by_epoch: dict[int, dict[str, Any]] = {}
    for r in rows:
        e = to_float(r.get("epoch"))
        if e is None:
            continue
        by_epoch[int(e)] = r
    return [by_epoch[k] for k in sorted(by_epoch)]


def series(rows: Iterable[dict[str, Any]], column: str) -> tuple[list[int], list[float]]:
    """(epochs, values) for one column, skipping epochs where it is absent."""
    xs, ys = [], []
    for r in rows:
        e, v = to_float(r.get("epoch")), to_float(r.get(column))
        if e is not None and v is not None:
            xs.append(int(e))
            ys.append(v)
    return xs, ys


def best_epoch(rows: Iterable[dict[str, Any]], column: str,
               mode: str = "max") -> tuple[int | None, float | None]:
    """Epoch and value of the best entry in `column`. mode: 'max' or 'min'."""
    xs, ys = series(rows, column)
    if not ys:
        return None, None
    idx = int(np.argmax(ys)) if mode == "max" else int(np.argmin(ys))
    return xs[idx], ys[idx]


# --------------------------------------------------------------------------
# Deterministic sample selection
# --------------------------------------------------------------------------

def select_image_ids(dataset_len: int, num_samples: int, seed: int) -> list[int]:
    """
    Choose evaluation image indices reproducibly.

    Depends ONLY on (dataset_len, num_samples, seed). It never sees a model, a
    checkpoint, or a metric, so it cannot be steered toward images that flatter
    one model -- and the identical list comes back for every model compared.

    Uses numpy's PCG64 explicitly rather than the global RNG, so an unrelated
    seeding call elsewhere in the process cannot change which images are picked.

    Selection is NESTED in num_samples: it draws one permutation and takes a
    prefix, so seed 42 with n=3 is a subset of seed 42 with n=8. Sampling
    independently per n would make the qualitative figure and the frequency
    figure silently use different images at the same seed, which is precisely
    the cherry-picking hazard this function exists to remove.
    """
    if num_samples <= 0:
        return []
    n = min(num_samples, dataset_len)
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.permutation(dataset_len)[:n])


# --------------------------------------------------------------------------
# Tensor <-> image helpers
# --------------------------------------------------------------------------

def to_numpy_image(t) -> np.ndarray:
    """
    (C,H,W) or (1,C,H,W) float tensor in [0,1] -> (H,W,C) float64 numpy, clipped.

    Clipping is applied because the model predicts a residual and can overshoot
    [0,1]; every figure must display the same clipped values the metrics use.
    """
    import torch
    if isinstance(t, torch.Tensor):
        t = t.detach().float().cpu()
        if t.dim() == 4:
            t = t[0]
        arr = t.permute(1, 2, 0).numpy()
    else:
        arr = np.asarray(t)
    return np.clip(arr, 0.0, 1.0)


# Error-map definitions. The paper must state which one a figure uses; the key
# used here is written into the figure provenance automatically.
ERROR_MODES = ("abs_mean", "squared_mean", "normalized")


def error_map(pred, gt, mode: str = "abs_mean") -> np.ndarray:
    """
    Pixel-wise restoration error between prediction and ground truth, in [0,1] units.

        abs_mean      E(x,y) = (1/3) * sum_c |I_pred_c - I_gt_c|
        squared_mean  E(x,y) = (1/3) * sum_c (I_pred_c - I_gt_c)^2
        normalized    abs_mean rescaled to [0,1] by ITS OWN max

    `normalized` is for visibility only. Its scale is per-image, so two
    normalized maps are NOT comparable to each other -- never use it to argue
    one model has lower error than another. Use abs_mean with a shared colour
    scale for that.
    """
    p = to_numpy_image(pred)
    g = to_numpy_image(gt)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs gt {g.shape}")

    if mode == "abs_mean":
        return np.abs(p - g).mean(axis=2)
    if mode == "squared_mean":
        return ((p - g) ** 2).mean(axis=2)
    if mode == "normalized":
        e = np.abs(p - g).mean(axis=2)
        m = float(e.max())
        return e / m if m > 0 else e
    raise ValueError(f"unknown error mode {mode!r}; expected one of {ERROR_MODES}")


def fft_log_magnitude(img) -> np.ndarray:
    """
    log(1 + |FFT2(I)|) of the luminance channel, zero frequency centred.

    Luminance (Rec.601) rather than per-channel so one spectrum represents the
    image; the three channels of a natural photo have near-identical spectra.
    """
    arr = to_numpy_image(img)
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    spectrum = np.fft.fftshift(np.fft.fft2(lum))
    return np.log1p(np.abs(spectrum))


def radial_profile(spectrum_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Azimuthally averaged profile of a centred 2-D spectrum.

    Returns (radius_in_pixels, mean_value_at_that_radius). Collapsing the 2-D
    spectrum this way makes "did high frequencies come back" legible as a curve
    rather than two grey squares a reader has to eyeball.

    Truncated at r = min(H,W)//2, the largest radius whose circle lies entirely
    inside the image. Past it a bin samples only the four corners, so it is not
    an azimuthal average at all; leaving those bins in produces a spurious
    upward spike at the right edge of the plot that invites exactly the
    "high frequencies were recovered" over-reading this module warns against.
    """
    h, w = spectrum_2d.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    nbins = int(r.max()) + 1
    total = np.bincount(r.ravel(), spectrum_2d.ravel(), minlength=nbins)
    count = np.bincount(r.ravel(), minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = np.where(count > 0, total / np.maximum(count, 1), np.nan)

    cutoff = min(h, w) // 2
    return np.arange(cutoff), prof[:cutoff]


def channel_mean_activation(feat) -> np.ndarray:
    """
    A(x,y) = (1/C) * sum_c |F_c(x,y)| for a (C,H,W) or (1,C,H,W) feature tensor.

    Mean of ABSOLUTE values: features are signed post-LayerNorm, so a plain mean
    cancels strong positive and negative responses into a misleading flat map.
    """
    import torch
    t = feat.detach().float().cpu()
    if t.dim() == 4:
        t = t[0]
    return t.abs().mean(dim=0).numpy()


def normalize01(a: np.ndarray) -> np.ndarray:
    """Rescale to [0,1] for display. Flat input maps to zeros, not NaN."""
    a = np.asarray(a, dtype=np.float64)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

ABLATION_ORDER = [
    "baseline_restormer", "seecoder", "seecoder_diffae",
    "seecoder_diffae_lcr", "seecoder_diffae_lcr_flow", "full_srrt",
]


# Results directories are not always named after the ablation preset: the
# baseline was written to results/restormer_baseline (words reversed) and the
# first ablation to results/ablation_seecoder. Without these, find_summary
# reported measured runs as "not run" and the ablation ladder rendered empty.
RESULTS_DIR_ALIASES: dict[str, tuple[str, ...]] = {
    "baseline_restormer": ("restormer_baseline",),
}

# Directory names that hold superseded runs and must never satisfy a lookup.
ARCHIVED_MARKERS = ("10ep", "_archive", "archive_", "_old", "superseded")


def _is_archived(path: str) -> bool:
    low = path.replace("\\", "/").lower()
    return any(m in low for m in ARCHIVED_MARKERS)


def find_summary(results_dir: str, name: str) -> str | None:
    """
    Locate an evaluate.py summary.json for one ablation, or None if unrun.

    Tried in order: the preset name, then known directory aliases, then a scan of
    `results_dir` for a summary.json whose recorded `experiment` field matches.
    The scan is the authoritative fallback -- it reads what the run declared
    itself to be rather than trusting a directory name.

    Archived directories (see ARCHIVED_MARKERS) are excluded, so the superseded
    10-epoch baseline can never be picked up as the current one.
    """
    candidates: list[str] = [name, f"ablation_{name}"]
    candidates += list(RESULTS_DIR_ALIASES.get(name, ()))

    for stem in candidates:
        for cand in (os.path.join(results_dir, stem, "summary.json"),
                     os.path.join(results_dir, stem, "metrics", "summary.json")):
            if os.path.exists(cand) and not _is_archived(cand):
                return cand

    # Fallback: ask each summary.json which experiment it belongs to.
    if not os.path.isdir(results_dir):
        return None
    matches: list[str] = []
    for entry in sorted(os.listdir(results_dir)):
        d = os.path.join(results_dir, entry)
        if not os.path.isdir(d) or _is_archived(d):
            continue
        for cand in (os.path.join(d, "summary.json"),
                     os.path.join(d, "metrics", "summary.json")):
            if not os.path.exists(cand):
                continue
            try:
                with io.open(cand, "r", encoding="utf-8") as f:
                    if json.load(f).get("experiment") == name:
                        matches.append(cand)
            except (OSError, ValueError):
                continue

    if not matches:
        return None
    if len(matches) > 1:
        # Ambiguity is a real problem, not something to resolve silently.
        matches.sort(key=os.path.getmtime, reverse=True)
        print(f"WARNING: {len(matches)} result dirs claim experiment {name!r}; "
              f"using the newest ({matches[0]}). Others: {matches[1:]}")
    return matches[0]


def load_ablation_results(results_dir: str = "results",
                          names: Sequence[str] | None = None) -> dict[str, dict | None]:
    """
    Load every ablation's measured summary.json.

    An ablation that has not been evaluated maps to None. Callers MUST render
    that as "not run" rather than substituting a value: an interpolated or
    carried-over number in an ablation table is fabricated evidence, and the
    whole point of the ladder is that each rung is measured.
    """
    out: dict[str, dict | None] = {}
    for name in (names or ABLATION_ORDER):
        path = find_summary(results_dir, name)
        if path is None:
            out[name] = None
            continue
        with io.open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_summary_path"] = path
        out[name] = data
    return out


def protocol_key(summary: dict) -> tuple:
    """
    The evaluation-protocol fingerprint of a result.

    Two results are only comparable if this matches: same split, same crop mode,
    same crop size, same image count. `assert_comparable` uses it to refuse to
    build a table mixing 256px centre crops with full-image evaluation, which
    would otherwise look like a clean several-dB improvement.
    """
    return (summary.get("split"), summary.get("crop_mode"),
            summary.get("crop_size"), summary.get("num_images"))


def assert_comparable(results: dict[str, dict | None]) -> tuple:
    """Raise unless every present result shares one evaluation protocol."""
    seen = {name: protocol_key(s) for name, s in results.items() if s}
    if not seen:
        return ()
    keys = set(seen.values())
    if len(keys) > 1:
        detail = "\n  ".join(f"{n}: split={k[0]} crop_mode={k[1]} crop_size={k[2]} "
                             f"n_images={k[3]}" for n, k in seen.items())
        raise SystemExit(
            "Refusing to combine results measured under different protocols:\n  "
            + detail +
            "\nRe-evaluate them the same way, or build separate tables. Crop-based "
            "and full-image numbers must never share a comparison table.")
    return next(iter(keys))


def load_checkpoint_model(checkpoint: str, device=None, config: str | None = None):
    """
    Rebuild the model a checkpoint was written from and load its weights.

    The checkpoint embeds its own config, so the ablation configuration is taken
    from the file rather than from whatever the YAML currently says. That is what
    lets a baseline figure and an SRRT figure be produced in one session without
    either being rebuilt with the wrong modules enabled.

    Returns (model in eval mode, cfg, meta) where meta carries epoch/best/params.
    """
    import torch
    import yaml
    from models.srrt import SRRT

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only=False: these checkpoints intentionally embed config + environment.
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    if config:
        with open(config, "r") as f:
            cfg = yaml.safe_load(f)
    elif "config" in ckpt:
        cfg = ckpt["config"]
    else:
        raise SystemExit(f"{checkpoint} has no stored config; pass --config.")

    model = SRRT.from_config(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    summary = model.module_summary()
    meta = {
        "checkpoint": checkpoint,
        "epoch": ckpt.get("epoch"),
        "best_val_psnr": ckpt.get("best_val_psnr"),
        "experiment": cfg["experiment"]["name"],
        "seed": cfg["experiment"]["seed"],
        "params_total": summary["total"],
        "modules_enabled": {k: summary[k]["enabled"] for k in
                            ("seecoder", "diffae", "lcr", "rectified_flow", "sra")},
    }
    return model, cfg, meta


def eval_dataset(cfg: dict[str, Any], split: str = "test", crop_mode: str | None = None,
                 crop_size: int | None = None):
    """
    Build the evaluation dataset exactly as evaluate.py does.

    Identical preprocessing across every model is a hard requirement of a fair
    comparison, so all figure scripts go through this one function rather than
    constructing datasets themselves.
    """
    from data.datasets import build_dataset

    val_cfg = cfg["dataset"]["val"]
    return build_dataset(
        name=cfg["dataset"].get("name", "gopro"),
        root=cfg["dataset"]["root"],
        split=split,
        crop_size=crop_size or val_cfg["crop_size"],
        crop_mode=crop_mode or val_cfg.get("crop_mode", "center"),
        augment=False,
    )
