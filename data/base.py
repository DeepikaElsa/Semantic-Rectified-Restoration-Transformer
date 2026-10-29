"""
Common base class for paired image-restoration datasets.

Every task (deblurring, deraining, dehazing, low-light, denoising, SR) reduces
to the same problem: a degraded image and its clean counterpart. Subclasses
only need to declare where the two directories live and what the task is
called, so train.py never needs task-specific branches.

Sample schema (identical for all datasets):
    {
        "input":    FloatTensor (3, H, W) in [0, 1] -- degraded image y
        "target":   FloatTensor (3, H, W) in [0, 1] -- clean image x
        "task":     str  -- e.g. "deblurring"
        "filename": str  -- input file name, for traceable qualitative output
    }
"""

import os
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from data.transforms import CropMode, PairedTransform

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _list_images(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Image directory does not exist: {directory}")

    names = sorted(
        f for f in os.listdir(directory)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )

    if not names:
        raise RuntimeError(f"No images found in {directory}")

    return names


class PairedRestorationDataset(Dataset):
    """
    Args:
        input_dir: directory of degraded images.
        target_dir: directory of clean images.
        task: task label stored on every sample.
        crop_size / crop_mode / augment: forwarded to PairedTransform.
        strict_pairing: if True (default), require input and target filenames
            to match exactly. Set False only for datasets whose paired files
            are named differently, and override `match_target` instead.

    Images are opened lazily in __getitem__; the dataset never holds decoded
    pixel data in RAM.
    """

    def __init__(
        self,
        input_dir: str,
        target_dir: str,
        task: str,
        crop_size: int = 128,
        crop_mode: CropMode = "random",
        augment: bool = True,
        strict_pairing: bool = True,
    ) -> None:
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.task = task

        input_names = _list_images(input_dir)
        target_names = set(_list_images(target_dir))

        # Validate correspondence explicitly. Matching list LENGTHS is not
        # sufficient -- two equally sized directories can still be misaligned,
        # which would train the model against wrong targets silently.
        self.pairs: list[tuple[str, str]] = []
        missing: list[str] = []

        for name in input_names:
            target_name = self.match_target(name, target_names)
            if target_name is None:
                missing.append(name)
            else:
                self.pairs.append((name, target_name))

        if strict_pairing and missing:
            preview = ", ".join(missing[:5])
            raise RuntimeError(
                f"{len(missing)} input image(s) in {input_dir} have no matching "
                f"target in {target_dir}. First few: {preview}"
            )

        if not self.pairs:
            raise RuntimeError(f"No valid input/target pairs between {input_dir} and {target_dir}")

        self.transform = PairedTransform(
            crop_size=crop_size,
            crop_mode=crop_mode,
            augment=augment,
        )

    def match_target(self, input_name: str, target_names: set[str]) -> str | None:
        """
        Map an input filename to its target filename.

        Default: identical names (GoPro, Rain100H, LOL-v2, SIDD...). Datasets
        with different naming (e.g. RESIDE hazy '0001_0.8_0.2.png' -> clean
        '0001.png') should override this.
        """
        return input_name if input_name in target_names else None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        input_name, target_name = self.pairs[idx]

        with Image.open(os.path.join(self.input_dir, input_name)) as im:
            inp = im.convert("RGB")
        with Image.open(os.path.join(self.target_dir, target_name)) as im:
            tgt = im.convert("RGB")

        x, y = self.transform(inp, tgt)

        return {
            "input": x,
            "target": y,
            "task": self.task,
            "filename": input_name,
        }
