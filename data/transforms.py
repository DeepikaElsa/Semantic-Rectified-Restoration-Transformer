"""
Paired image transforms for restoration.

CRITICAL: degraded/clean pairs must receive the IDENTICAL geometric transform.
Applying torchvision transforms to each image separately would desynchronize
the pair (a random crop would sample different regions), silently destroying
the supervision signal. Every transform here takes and returns both images.

Randomness uses torch's RNG rather than Python's `random`, because torch seeds
its RNG per DataLoader worker automatically -- with `random` the workers can
draw correlated crops on some platforms.
"""

from typing import Literal

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

CropMode = Literal["random", "center", "none"]


def _pad(x: torch.Tensor, pad_w: int, pad_h: int) -> torch.Tensor:
    """
    Pad a (C, H, W) tensor on the right/bottom, avoiding black borders.

    Prefers reflection padding, but torch requires the pad amount to be
    strictly smaller than the corresponding input dimension; when that does
    not hold (an image smaller than the target size) it falls back to
    replication, which has no such constraint.
    """
    if pad_h == 0 and pad_w == 0:
        return x

    _, h, w = x.shape
    mode = "reflect" if (pad_h < h and pad_w < w) else "replicate"

    return F.pad(x.unsqueeze(0), (0, pad_w, 0, pad_h), mode=mode).squeeze(0)


def pad_to_multiple(x: torch.Tensor, multiple: int = 8) -> torch.Tensor:
    """
    Pad a (C, H, W) tensor so H and W are divisible by `multiple`.

    The Restormer encoder downsamples three times, so inputs must be divisible
    by 8.
    """
    _, h, w = x.shape
    return _pad(x, (multiple - w % multiple) % multiple, (multiple - h % multiple) % multiple)


class PairedTransform:
    """
    Applies matched geometric augmentation + tensor conversion to an
    (input, target) image pair.

    Args:
        crop_size: square crop side length. Ignored when crop_mode='none'.
        crop_mode:
            'random' -- random crop (training). Paired, same coordinates.
            'center' -- deterministic center crop (validation), so metrics are
                        comparable across epochs.
            'none'   -- keep full image, reflection-padded to a multiple of 8.
        augment: enable random horizontal/vertical flips and 90-degree
                 rotations. Should be True only for training.
        size_multiple: divisibility requirement imposed by the backbone.

    Returns:
        (input_tensor, target_tensor), both float32 in [0, 1], shape (3, H, W).
    """

    def __init__(
        self,
        crop_size: int = 128,
        crop_mode: CropMode = "random",
        augment: bool = True,
        size_multiple: int = 8,
    ) -> None:
        if crop_mode not in ("random", "center", "none"):
            raise ValueError(f"crop_mode must be 'random', 'center' or 'none', got {crop_mode!r}")
        if crop_mode != "none" and crop_size % size_multiple != 0:
            raise ValueError(f"crop_size ({crop_size}) must be divisible by {size_multiple}")

        self.crop_size = crop_size
        self.crop_mode = crop_mode
        self.augment = augment
        self.size_multiple = size_multiple

    def _pad_if_smaller(self, x: torch.Tensor) -> torch.Tensor:
        """Pad up to crop_size when an image is smaller than the crop."""
        _, h, w = x.shape
        return _pad(x, max(0, self.crop_size - w), max(0, self.crop_size - h))

    def __call__(self, inp: Image.Image, tgt: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        x = TF.to_tensor(inp)
        y = TF.to_tensor(tgt)

        if x.shape != y.shape:
            raise ValueError(f"input/target shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")

        if self.crop_mode == "none":
            return pad_to_multiple(x, self.size_multiple), pad_to_multiple(y, self.size_multiple)

        x = self._pad_if_smaller(x)
        y = self._pad_if_smaller(y)

        _, h, w = x.shape
        size = self.crop_size

        if self.crop_mode == "random":
            top = int(torch.randint(0, h - size + 1, (1,)).item())
            left = int(torch.randint(0, w - size + 1, (1,)).item())
        else:  # center
            top = (h - size) // 2
            left = (w - size) // 2

        x = x[:, top:top + size, left:left + size]
        y = y[:, top:top + size, left:left + size]

        if self.augment:
            if torch.rand(1).item() < 0.5:                      # horizontal flip
                x, y = torch.flip(x, dims=[2]), torch.flip(y, dims=[2])
            if torch.rand(1).item() < 0.5:                      # vertical flip
                x, y = torch.flip(x, dims=[1]), torch.flip(y, dims=[1])
            k = int(torch.randint(0, 4, (1,)).item())           # 0/90/180/270 rotation
            if k:
                x, y = torch.rot90(x, k, dims=[1, 2]), torch.rot90(y, k, dims=[1, 2])

        return x.contiguous(), y.contiguous()
