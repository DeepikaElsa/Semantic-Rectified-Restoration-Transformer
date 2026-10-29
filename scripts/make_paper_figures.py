"""
Generate every data-driven figure in paper/srrt_ieee.tex.

Writes PDF (vector, for LaTeX) and PNG (for quick viewing) into paper/figures/.
Each figure also writes a .json sidecar recording exactly which artifacts it was
built from, so no number in the paper is untraceable.

    python scripts/make_paper_figures.py              # all figures
    python scripts/make_paper_figures.py --skip-qual  # plots only (no GPU)

Figures
    fig_convergence  validation PSNR vs epoch, all six arms
    fig_ablation     final PSNR / LPIPS per arm, baseline-relative
    fig_divergence   pre-fix vs post-fix rectified-flow training
    fig_qualitative  blurred / baseline / SRRT / ground truth, with zoom insets
"""

import argparse
import csv
import json
import os
import sys
from typing import Any

# Same convention as the other scripts in this directory: make the project
# root importable so `data.` and `models.` resolve when run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join("paper", "figures")

# Arms in ladder order, with the short labels used in the paper's tables.
ARMS = [
    ("baseline_restormer",       "A: Restormer",      "#4C4C4C"),
    ("seecoder",                 "B: +SeeCoder",      "#0072B2"),
    ("seecoder_diffae",          "C: +DiffAE",        "#009E73"),
    ("seecoder_diffae_lcr",      "D: +LCR",           "#E69F00"),
    ("seecoder_diffae_lcr_flow", "E: +Flow",          "#D55E00"),
    ("full_srrt",                "F: +SRA (full)",    "#CC79A7"),
]
RESULT_DIR = {
    "baseline_restormer": "restormer_baseline",   # named by hand before the chain existed
}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# IEEE two-column widths, in inches.
COL, FULL = 3.5, 7.16


def save(fig, name: str, provenance: dict[str, Any]) -> None:
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(provenance, f, indent=2)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def read_log(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r.get("epoch", "").isdigit()]


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """train_log.csv is appended across resumes, so an epoch can appear twice.
    The LAST occurrence is the one that actually stands."""
    by_epoch: dict[int, dict[str, str]] = {}
    for r in rows:
        by_epoch[int(r["epoch"])] = r
    return [by_epoch[e] for e in sorted(by_epoch)]


def split_runs(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    """Split an appended log into separate runs wherever the epoch counter
    restarts at 0."""
    runs, cur = [], []
    for r in rows:
        if int(r["epoch"]) == 0 and cur:
            runs.append(cur)
            cur = []
        cur.append(r)
    if cur:
        runs.append(cur)
    return runs


def summary(name: str) -> dict[str, Any] | None:
    d = RESULT_DIR.get(name, name)
    p = os.path.join("results", d, "summary.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


# ------------------------------------------------------------------ figure 1
def fig_convergence() -> None:
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    prov: dict[str, Any] = {"source": {}}
    for name, label, color in ARMS:
        rows = dedupe(read_log(os.path.join("logs", name, "train_log.csv")))
        if not rows:
            continue
        ep = [int(r["epoch"]) for r in rows]
        ps = [float(r["val_psnr"]) for r in rows]
        lw = 1.4 if name == "seecoder_diffae_lcr_flow" else 0.9
        z = 3 if name == "seecoder_diffae_lcr_flow" else 2
        ax.plot(ep, ps, label=label, color=color, linewidth=lw, zorder=z)
        prov["source"][name] = f"logs/{name}/train_log.csv"
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation PSNR (dB)")
    ax.set_ylim(22.5, 29.9)
    ax.set_xlim(0, 59)
    ax.legend(loc="lower right", ncol=2, frameon=False, handlelength=1.4,
              columnspacing=0.9, borderpad=0.2)

    # The arms separate by ~0.1 dB on a 6 dB axis, so on the main plot they lie
    # on top of one another. The inset is where the ablation is actually
    # legible -- without it the figure would imply the arms are identical.
    ins = ax.inset_axes([0.085, 0.60, 0.42, 0.37])
    for name, label, color in ARMS:
        rows = dedupe(read_log(os.path.join("logs", name, "train_log.csv")))
        if not rows:
            continue
        sel = [r for r in rows if int(r["epoch"]) >= 45]
        lw = 1.3 if name == "seecoder_diffae_lcr_flow" else 0.8
        ins.plot([int(r["epoch"]) for r in sel],
                 [float(r["val_psnr"]) for r in sel], color=color, linewidth=lw)
    ins.set_xlim(45, 59)
    ins.set_ylim(28.55, 29.15)
    ins.tick_params(labelsize=5.5, length=2, pad=1)
    ins.set_title("final 15 epochs", fontsize=5.8, pad=2)
    ins.grid(alpha=0.25, linewidth=0.3)
    for s in ins.spines.values():
        s.set_linewidth(0.5)
        s.set_visible(True)
    save(fig, "fig_convergence", prov)


# ------------------------------------------------------------------ figure 2
def fig_ablation() -> None:
    labels, psnr, lpips, names = [], [], [], []
    for name, label, _ in ARMS:
        s = summary(name)
        if not s:
            continue
        labels.append(label.split(": ")[1])
        psnr.append(s["metrics"]["psnr"]["mean"])
        lpips.append(s["metrics"]["lpips"]["mean"])
        names.append(name)
    base_p, base_l = psnr[0], lpips[0]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 1.95))
    x = np.arange(len(labels))

    dp = [p - base_p for p in psnr]
    c1 = ["#4C4C4C" if i == 0 else ("#2E7D32" if v > 0 else "#B71C1C")
          for i, v in enumerate(dp)]
    a1.bar(x, dp, color=c1, width=0.62)
    a1.axhline(0, color="black", linewidth=0.7)
    a1.set_ylabel(r"$\Delta$PSNR vs baseline (dB)")
    for i, (v, p) in enumerate(zip(dp, psnr)):
        a1.text(i, v + (0.006 if v >= 0 else -0.016), f"{p:.3f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=6.2)
    a1.set_ylim(-0.045, 0.165)

    dl = [base_l - l for l in lpips]          # positive = better (LPIPS is lower-better)
    c2 = ["#4C4C4C" if i == 0 else ("#2E7D32" if v > 0 else "#B71C1C")
          for i, v in enumerate(dl)]
    a2.bar(x, dl, color=c2, width=0.62)
    a2.axhline(0, color="black", linewidth=0.7)
    a2.set_ylabel(r"LPIPS improvement vs baseline")
    for i, (v, l) in enumerate(zip(dl, lpips)):
        a2.text(i, v + (0.0004 if v >= 0 else -0.0009), f"{l:.4f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=6.2)

    for a in (a1, a2):
        a.set_xticks(x)
        a.set_xticklabels(labels, rotation=18, ha="right")
        a.grid(axis="x", visible=False)
    save(fig, "fig_ablation",
         {"source": {n: f"results/{RESULT_DIR.get(n, n)}/summary.json" for n in names}})


# ------------------------------------------------------------------ figure 3
def fig_divergence() -> None:
    """The pre-fix flow run and the corrected one share an appended log; the
    failed run is the first block, before the epoch counter restarts."""
    acc = read_log("logs/seecoder_diffae_lcr_flow_200EP_ACCIDENTAL/train_log.csv")
    runs = split_runs(acc)
    broken = runs[0] if runs else []
    fixed = dedupe(read_log("logs/seecoder_diffae_lcr_flow/train_log.csv"))
    lcr = dedupe(read_log("logs/seecoder_diffae_lcr/train_log.csv"))

    fig, ax = plt.subplots(figsize=(COL, 2.1))
    n = 20
    if broken:
        ax.plot([int(r["epoch"]) for r in broken][:n],
                [float(r["val_psnr"]) for r in broken][:n],
                color="#B71C1C", linewidth=1.3, marker="o", markersize=2.4,
                label="E, pre-fix (diverges)")
    if lcr:
        ax.plot([int(r["epoch"]) for r in lcr][:n],
                [float(r["val_psnr"]) for r in lcr][:n],
                color="#9E9E9E", linewidth=1.0, linestyle="--",
                label="D: +LCR (no flow)")
    if fixed:
        ax.plot([int(r["epoch"]) for r in fixed][:n],
                [float(r["val_psnr"]) for r in fixed][:n],
                color="#2E7D32", linewidth=1.3, marker="s", markersize=2.4,
                label="E, post-fix")
    ax.axvspan(5.5, 6.5, color="#B71C1C", alpha=0.10, linewidth=0)
    ax.annotate("collapse at\nepoch 6", xy=(6.1, 14.3), xytext=(8.6, 19.6),
                fontsize=6.4, color="#B71C1C", ha="center",
                arrowprops=dict(arrowstyle="->", color="#B71C1C", linewidth=0.7))
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation PSNR (dB)")
    ax.set_xlim(0, n - 1)
    ax.set_xticks(range(0, n, 4))
    ax.set_ylim(5, 27.5)
    ax.legend(loc="lower left", frameon=False, handlelength=1.6, borderpad=0.2)
    save(fig, "fig_divergence", {"source": {
        "pre_fix":  "logs/seecoder_diffae_lcr_flow_200EP_ACCIDENTAL/train_log.csv (first run block)",
        "post_fix": "logs/seecoder_diffae_lcr_flow/train_log.csv",
        "lcr":      "logs/seecoder_diffae_lcr/train_log.csv"}})


# ------------------------------------------------------------------ figure 4
def fig_qualitative(images: list[str], crop: int, zoom: int) -> None:
    """Blurred / baseline / SRRT / ground truth, each with a magnified inset --
    the layout used throughout the deblurring literature."""
    import torch
    import yaml
    from data.datasets import create_dataloader
    from models.srrt import SRRT

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(ckpt_path: str, wanted: set[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        model = SRRT.from_config(cfg).to(dev).eval()
        model.load_state_dict(ck["model_state"])
        loader = create_dataloader(
            root=cfg["dataset"]["root"], name=cfg["dataset"].get("name", "gopro"),
            split="test", batch_size=1, crop_size=crop, crop_mode="center",
            augment=False, shuffle=False, num_workers=0, drop_last=False)
        out, inp, tgt = {}, {}, {}
        with torch.no_grad():
            for b in loader:
                fn = b["filename"][0] if isinstance(b["filename"], list) else b["filename"]
                if fn not in wanted:
                    continue
                x = b["input"].to(dev)
                y = model(x)
                y = y["output"] if isinstance(y, dict) else y
                out[fn] = y.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                inp[fn] = b["input"][0].permute(1, 2, 0).numpy()
                tgt[fn] = b["target"][0].permute(1, 2, 0).numpy()
                if len(out) == len(wanted):
                    break
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return out, inp, tgt

    want = set(images)
    print("  running baseline ...")
    base, inp, tgt = run("checkpoints/baseline_restormer/best_psnr.pth", want)
    print("  running SRRT (E) ...")
    srrt, _, _ = run("checkpoints/seecoder_diffae_lcr_flow/best_psnr.pth", want)

    def psnr(a: np.ndarray, b: np.ndarray) -> float:
        m = float(np.mean((a - b) ** 2))
        return 99.0 if m <= 0 else 10.0 * float(np.log10(1.0 / m))

    cols = ["Blurred input", "Restormer (A)", "SRRT (E)", "Ground truth"]
    rows = [f for f in images if f in base]

    # Two rows per test image: the FULL restored frames, then the magnified
    # detail. Showing only crops hides what the model actually outputs; showing
    # only full frames hides the differences, which are sub-pixel at this size.
    fig, axes = plt.subplots(2 * len(rows), 4,
                             figsize=(FULL, 1.80 * len(rows) * 2 * 0.56),
                             gridspec_kw={"wspace": 0.02, "hspace": 0.04})
    if len(rows) == 1:
        axes = axes.reshape(2, 4)

    for r, fn in enumerate(rows):
        # Choose the zoom window where SRRT improves on the baseline most.
        # Selecting on maximum BASELINE error instead lands on regions so
        # destroyed that neither method recovers anything, which shows the
        # reader nothing. This selection is favourable by construction and is
        # disclosed as such in the caption; the unselected full-test-set
        # averages are the numbers in Table II.
        from numpy.lib.stride_tricks import sliding_window_view
        k = zoom
        gain = (np.mean((base[fn] - tgt[fn]) ** 2, axis=2)
                - np.mean((srrt[fn] - tgt[fn]) ** 2, axis=2))
        win = sliding_window_view(gain, (k, k)).mean(axis=(2, 3))
        cy, cx = np.unravel_index(np.argmax(win), win.shape)
        sl = (slice(cy, cy + k), slice(cx, cx + k))

        rf, rz = 2 * r, 2 * r + 1          # full-frame row, zoom row
        full = [inp[fn], base[fn], srrt[fn], tgt[fn]]
        patch = [im[sl] for im in full]   # NB: do not name this 'crop' -- shadows the crop-size arg

        # Full-image PSNR on the top row, patch PSNR on the magnified row --
        # a number printed on a panel has to describe that panel.
        sf = [psnr(inp[fn], tgt[fn]), psnr(base[fn], tgt[fn]),
              psnr(srrt[fn], tgt[fn]), None]
        sz = [psnr(patch[0], patch[3]), psnr(patch[1], patch[3]),
              psnr(patch[2], patch[3]), None]

        for c in range(4):
            a = axes[rf, c]
            a.imshow(np.clip(full[c], 0, 1))
            a.add_patch(plt.Rectangle((cx, cy), k, k, fill=False,
                                      edgecolor="#FF1744", linewidth=0.9))
            if r == 0:
                a.set_title(cols[c], fontsize=7, pad=3)
            if sf[c] is not None:
                a.text(0.025, 0.035, f"{sf[c]:.2f} dB", transform=a.transAxes,
                       fontsize=5.8, color="white", va="bottom",
                       bbox=dict(facecolor="black", alpha=0.55, pad=1.0,
                                 edgecolor="none"))
            b = axes[rz, c]
            b.imshow(np.clip(patch[c], 0, 1))
            if sz[c] is not None:
                b.text(0.025, 0.035, f"{sz[c]:.2f} dB", transform=b.transAxes,
                       fontsize=5.8, color="white", va="bottom",
                       bbox=dict(facecolor="black", alpha=0.55, pad=1.0,
                                 edgecolor="none"))

        axes[rf, 0].set_ylabel(fn.replace(".png", "") + "\nfull", fontsize=6.2)
        axes[rz, 0].set_ylabel("detail", fontsize=6.2)
        for rr in (rf, rz):
            for c in range(4):
                axes[rr, c].set_xticks([])
                axes[rr, c].set_yticks([])
                axes[rr, c].grid(False)

    save(fig, "fig_qualitative", {
        "images": rows, "crop": crop, "zoom": zoom,
        "zoom_selection": "window maximising baseline-minus-SRRT squared error (favourable by construction; disclosed in caption)",
        "checkpoints": ["checkpoints/baseline_restormer/best_psnr.pth",
                        "checkpoints/seecoder_diffae_lcr_flow/best_psnr.pth"]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-qual", action="store_true", help="plots only, no GPU needed")
    ap.add_argument("--images", nargs="+", default=["157.png", "486.png"])
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--zoom", type=int, default=128)
    a = ap.parse_args()

    print("figures -> paper/figures/")
    fig_convergence()
    fig_ablation()
    fig_divergence()
    if not a.skip_qual:
        fig_qualitative(a.images, a.crop, a.zoom)
    print("done.")


if __name__ == "__main__":
    main()
