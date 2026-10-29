"""
What in the pixels actually signals "blurred" vs "scratched"?

Measures, on real GoPro pairs, which image statistics separate a degraded image
from its clean counterpart -- and which do not. Also synthesizes a scratch to
contrast the two degradation signatures.

This is an ANALYSIS tool. It changes no architecture and trains nothing; it
exists to ground the design of a damage-aware encoder (DiffAE) in measured
properties of the data rather than intuition.

    python scripts/analyze_degradation_cues.py --num-images 50
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import GoProDataset  # noqa: E402

# 3x3 Laplacian: responds to second spatial derivative, i.e. how much a pixel
# deviates from the average of its neighbours.
LAPLACIAN = torch.tensor([[0., 1., 0.],
                          [1., -4., 1.],
                          [0., 1., 0.]]).view(1, 1, 3, 3)

SOBEL_X = torch.tensor([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]]).view(1, 1, 3, 3)
SOBEL_Y = SOBEL_X.transpose(-1, -2).contiguous()


def gray(img: torch.Tensor) -> torch.Tensor:
    """(3,H,W) in [0,1] -> (1,1,H,W) luminance."""
    w = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    return (img * w).sum(0)[None, None]


def laplacian_variance(g: torch.Tensor) -> float:
    """Classic sharpness metric: variance of the Laplacian response."""
    return float(F.conv2d(g, LAPLACIAN).var())


def gradient_magnitude(g: torch.Tensor) -> float:
    gx = F.conv2d(g, SOBEL_X)
    gy = F.conv2d(g, SOBEL_Y)
    return float(torch.sqrt(gx ** 2 + gy ** 2).mean())


def high_freq_ratio(g: torch.Tensor) -> float:
    """Fraction of FFT energy outside the central (low-frequency) quarter."""
    f = torch.fft.fftshift(torch.fft.fft2(g[0, 0]))
    power = (f.real ** 2 + f.imag ** 2)
    h, w = power.shape
    ch, cw = h // 2, w // 2
    rh, rw = h // 8, w // 8
    low = power[ch - rh:ch + rh, cw - rw:cw + rw].sum()
    return float((power.sum() - low) / power.sum().clamp(min=1e-12))


def neighbour_correlation(g: torch.Tensor) -> float:
    """Pearson correlation between each pixel and its right-hand neighbour."""
    a = g[0, 0, :, :-1].flatten()
    b = g[0, 0, :, 1:].flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-12))


def extreme_gradient_fraction(g: torch.Tensor, k: float = 8.0) -> float:
    """
    Fraction of pixels whose gradient exceeds k standard deviations.

    Sparse, extreme outliers are the signature of a scratch; blur has none.
    """
    gx = F.conv2d(g, SOBEL_X)
    gy = F.conv2d(g, SOBEL_Y)
    mag = torch.sqrt(gx ** 2 + gy ** 2)
    return float((mag > mag.mean() + k * mag.std()).float().mean())


def add_synthetic_scratch(img: torch.Tensor, n: int = 3, seed: int = 0) -> torch.Tensor:
    """
    Draw thin, bright, near-straight lines -- a first-order stand-in for
    physical film scratches. GoPro contains NO scratches, so this exists purely
    to contrast the two degradation signatures.
    """
    g = torch.Generator().manual_seed(seed)
    out = img.clone()
    _, h, w = out.shape
    for _ in range(n):
        x = int(torch.randint(0, w, (1,), generator=g))
        thickness = int(torch.randint(1, 3, (1,), generator=g))
        drift = float(torch.empty(1).uniform_(-0.15, 0.15, generator=g))
        value = float(torch.empty(1).uniform_(0.85, 1.0, generator=g))
        for y in range(h):
            xx = int(x + drift * y)
            if 0 <= xx < w - thickness:
                out[:, y, xx:xx + thickness] = value
    return out


def stats(img: torch.Tensor) -> dict:
    g = gray(img)
    return {
        "mean_intensity": float(img.mean()),
        "std_intensity": float(img.std()),
        "laplacian_var": laplacian_variance(g),
        "grad_magnitude": gradient_magnitude(g),
        "high_freq_ratio": high_freq_ratio(g),
        "neighbour_corr": neighbour_correlation(g),
        "extreme_grad_frac": extreme_gradient_fraction(g),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Gopro")
    ap.add_argument("--num-images", type=int, default=50)
    ap.add_argument("--crop", type=int, default=256)
    args = ap.parse_args()

    ds = GoProDataset(root=args.root, split="train", crop_size=args.crop,
                      crop_mode="center", augment=False)
    n = min(args.num_images, len(ds))

    acc = {"blurred": [], "sharp": [], "sharp+scratch": []}
    for i in range(n):
        s = ds[i]
        acc["blurred"].append(stats(s["input"]))
        acc["sharp"].append(stats(s["target"]))
        acc["sharp+scratch"].append(stats(add_synthetic_scratch(s["target"], seed=i)))

    keys = list(acc["sharp"][0].keys())
    mean = {c: {k: float(np.mean([d[k] for d in acc[c]])) for k in keys} for c in acc}

    print("=" * 88)
    print(f"DEGRADATION CUES -- {n} GoPro pairs, {args.crop}px center crops")
    print("=" * 88)
    print(f"{'statistic':<22}{'sharp':>14}{'blurred':>14}{'sharp+scratch':>16}"
          f"{'blur/sharp':>12}")
    print("-" * 88)
    for k in keys:
        s_v, b_v, c_v = mean["sharp"][k], mean["blurred"][k], mean["sharp+scratch"][k]
        ratio = b_v / s_v if s_v not in (0.0,) else float("nan")
        print(f"{k:<22}{s_v:>14.5f}{b_v:>14.5f}{c_v:>16.5f}{ratio:>11.2f}x")
    print("-" * 88)

    # Separability: how reliably does each statistic order the pair
    # (blurred vs its own sharp target) the expected way, per image?
    print("\nPer-image discriminability (blurred vs its OWN sharp target):")
    for k in keys:
        b = np.array([d[k] for d in acc["blurred"]])
        s = np.array([d[k] for d in acc["sharp"]])
        agree = float(np.mean(b < s))          # fraction where blurred < sharp
        direction = "blurred LOWER" if agree >= 0.5 else "blurred HIGHER"
        acc_pct = max(agree, 1 - agree) * 100
        print(f"  {k:<22} {direction:<16} correct on {acc_pct:5.1f}% of images")

    print("\nScratch signature (sharp vs sharp+scratch), same image:")
    for k in ("laplacian_var", "grad_magnitude", "high_freq_ratio", "extreme_grad_frac"):
        s_v, c_v = mean["sharp"][k], mean["sharp+scratch"][k]
        arrow = "UP" if c_v > s_v else "DOWN"
        print(f"  {k:<22} {s_v:>12.5f} -> {c_v:>12.5f}   ({arrow})")

    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
