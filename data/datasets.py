"""
Dataset registry and DataLoader construction.

Only GoPro is implemented for now, by design: the framework is validated on a
single task (motion deblurring) before the remaining benchmarks are added. The
registry below is the extension point -- adding a dataset means adding one
subclass and one registry entry, with no change to train.py or evaluate.py.

Planned (not yet implemented):
    Rain100H -> deraining, RESIDE -> dehazing, LOL-v2 -> low-light,
    SIDD -> denoising, DIV2K -> super-resolution.
"""

import os
from typing import Any, Type

from torch.utils.data import DataLoader, Dataset

from data.base import PairedRestorationDataset
from data.transforms import CropMode


class GoProDataset(PairedRestorationDataset):
    """
    GoPro motion-deblurring dataset.

    Expected layout (matches this repository):
        <root>/train/blur/*.png   <root>/train/sharp/*.png
        <root>/test/blur/*.png    <root>/test/sharp/*.png

    Blur and sharp frames share filenames, so the default exact-name matching
    in PairedRestorationDataset applies.
    """

    task_name = "deblurring"

    def __init__(
        self,
        root: str,
        split: str = "train",
        crop_size: int = 128,
        crop_mode: CropMode = "random",
        augment: bool = True,
    ) -> None:
        super().__init__(
            input_dir=os.path.join(root, split, "blur"),
            target_dir=os.path.join(root, split, "sharp"),
            task=self.task_name,
            crop_size=crop_size,
            crop_mode=crop_mode,
            augment=augment,
        )


DATASET_REGISTRY: dict[str, Type[PairedRestorationDataset]] = {
    "gopro": GoProDataset,
}

PLANNED_DATASETS = ("rain100h", "reside", "lol-v2", "sidd", "div2k")


def build_dataset(
    name: str,
    root: str,
    split: str = "train",
    crop_size: int = 128,
    crop_mode: CropMode = "random",
    augment: bool = True,
) -> Dataset:
    """Instantiate a registered dataset by (case-insensitive) name."""
    key = name.lower()

    if key not in DATASET_REGISTRY:
        if key in PLANNED_DATASETS:
            raise NotImplementedError(
                f"Dataset {name!r} is planned but not implemented yet. "
                f"Currently available: {sorted(DATASET_REGISTRY)}"
            )
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(DATASET_REGISTRY)}")

    return DATASET_REGISTRY[key](
        root=root,
        split=split,
        crop_size=crop_size,
        crop_mode=crop_mode,
        augment=augment,
    )


def create_dataloader(
    root: str,
    name: str = "gopro",
    split: str = "train",
    batch_size: int = 2,
    crop_size: int = 128,
    crop_mode: CropMode = "random",
    augment: bool = True,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Build a DataLoader over a registered dataset.

    Windows note: DataLoader workers use the 'spawn' start method, so any
    script calling this MUST be guarded by `if __name__ == "__main__":`.
    `persistent_workers` avoids paying the (expensive) respawn cost every
    epoch when workers are enabled.
    """
    dataset = build_dataset(
        name=name,
        root=root,
        split=split,
        crop_size=crop_size,
        crop_mode=crop_mode,
        augment=augment,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def dataloader_from_config(cfg: dict[str, Any], split: str = "train") -> DataLoader:
    """
    Build a DataLoader straight from a loaded YAML config.

    Reads `dataset.root`, `dataset.name` and the per-split block
    `dataset.<split>` (batch_size, crop_size, crop_mode, augment, shuffle,
    num_workers, drop_last).
    """
    ds_cfg = cfg["dataset"]
    split_cfg = ds_cfg[split]

    return create_dataloader(
        root=ds_cfg["root"],
        name=ds_cfg.get("name", "gopro"),
        split=split_cfg.get("split", split),
        batch_size=split_cfg["batch_size"],
        crop_size=split_cfg["crop_size"],
        crop_mode=split_cfg.get("crop_mode", "random" if split == "train" else "center"),
        augment=split_cfg.get("augment", split == "train"),
        shuffle=split_cfg.get("shuffle", split == "train"),
        num_workers=split_cfg.get("num_workers", 0),
        drop_last=split_cfg.get("drop_last", split == "train"),
    )
