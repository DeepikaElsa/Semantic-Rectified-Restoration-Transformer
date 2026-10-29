import pytest
import torch
import yaml
from PIL import Image

from data.base import PairedRestorationDataset
from data.datasets import GoProDataset, build_dataset, create_dataloader, dataloader_from_config
from data.transforms import PairedTransform, pad_to_multiple

GOPRO_ROOT = "data/Gopro"


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------

def _pair(w=64, h=48):
    """Two distinguishable images: a gradient and its inverse."""
    a = Image.new("RGB", (w, h))
    a.putdata([(x % 256, y % 256, 0) for y in range(h) for x in range(w)])
    b = Image.new("RGB", (w, h))
    b.putdata([(255 - x % 256, 255 - y % 256, 0) for y in range(h) for x in range(w)])
    return a, b


def test_pad_to_multiple():
    assert pad_to_multiple(torch.randn(3, 30, 45), 8).shape == (3, 32, 48)
    assert pad_to_multiple(torch.randn(3, 32, 48), 8).shape == (3, 32, 48)


def test_random_crop_is_paired():
    """
    The SAME crop window must be taken from input and target. Cropping them
    independently would silently destroy the supervision signal, so this is
    checked structurally rather than by shape alone.
    """
    img = Image.new("RGB", (64, 64))
    img.putdata([(x, y, 0) for y in range(64) for x in range(64)])

    tf = PairedTransform(crop_size=16, crop_mode="random", augment=False)

    for _ in range(20):
        x, y = tf(img, img)
        assert x.shape == y.shape == (3, 16, 16)
        # Identical inputs must stay identical after a *paired* transform.
        assert torch.equal(x, y)


def test_center_crop_is_deterministic():
    a, b = _pair()
    tf = PairedTransform(crop_size=32, crop_mode="center", augment=False)

    first = tf(a, b)
    for _ in range(5):
        again = tf(a, b)
        assert torch.equal(first[0], again[0])
        assert torch.equal(first[1], again[1])


def test_crop_mode_none_pads_to_multiple_of_8():
    a, b = _pair(w=45, h=30)
    x, y = PairedTransform(crop_mode="none", augment=False)(a, b)
    assert x.shape == y.shape == (3, 32, 48)


def test_pad_if_smaller_than_crop():
    a, b = _pair(w=20, h=12)
    x, y = PairedTransform(crop_size=32, crop_mode="center", augment=False)(a, b)
    assert x.shape == y.shape == (3, 32, 32)


def test_augmentation_keeps_pair_aligned():
    img = Image.new("RGB", (32, 32))
    img.putdata([(x, y, 0) for y in range(32) for x in range(32)])

    tf = PairedTransform(crop_size=16, crop_mode="random", augment=True)
    for _ in range(20):
        x, y = tf(img, img)
        assert torch.equal(x, y)


def test_transform_rejects_bad_crop_mode():
    with pytest.raises(ValueError):
        PairedTransform(crop_mode="diagonal")


def test_transform_rejects_crop_not_divisible_by_8():
    with pytest.raises(ValueError):
        PairedTransform(crop_size=30, crop_mode="random")


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------

def test_gopro_sample_schema():
    ds = GoProDataset(root=GOPRO_ROOT, split="train", crop_size=64)
    sample = ds[0]

    assert set(sample) == {"input", "target", "task", "filename"}
    assert sample["input"].shape == (3, 64, 64)
    assert sample["target"].shape == (3, 64, 64)
    assert sample["input"].dtype == torch.float32
    assert sample["task"] == "deblurring"
    assert isinstance(sample["filename"], str)

    for key in ("input", "target"):
        t = sample[key]
        assert torch.isfinite(t).all()
        assert 0.0 <= t.min() and t.max() <= 1.0


def test_gopro_splits_have_expected_sizes():
    assert len(GoProDataset(root=GOPRO_ROOT, split="train", crop_size=64)) == 2103
    assert len(GoProDataset(root=GOPRO_ROOT, split="test", crop_size=64)) == 1111


def test_pairing_validation_rejects_unmatched(tmp_path):
    """A target directory missing a file must raise, not silently misalign."""
    inp = tmp_path / "blur"
    tgt = tmp_path / "sharp"
    inp.mkdir()
    tgt.mkdir()

    for name in ("a.png", "b.png"):
        Image.new("RGB", (32, 32)).save(inp / name)
    Image.new("RGB", (32, 32)).save(tgt / "a.png")   # 'b.png' deliberately absent

    with pytest.raises(RuntimeError, match="no matching target"):
        PairedRestorationDataset(str(inp), str(tgt), task="test", crop_size=16)


def test_missing_directory_raises():
    with pytest.raises(FileNotFoundError):
        GoProDataset(root="data/DoesNotExist", split="train")


def test_build_dataset_registry():
    assert isinstance(build_dataset("gopro", GOPRO_ROOT, "train", crop_size=64), GoProDataset)
    assert isinstance(build_dataset("GoPro", GOPRO_ROOT, "train", crop_size=64), GoProDataset)

    with pytest.raises(NotImplementedError):
        build_dataset("rain100h", GOPRO_ROOT, "train")

    with pytest.raises(KeyError):
        build_dataset("not_a_dataset", GOPRO_ROOT, "train")


# --------------------------------------------------------------------------
# dataloader
# --------------------------------------------------------------------------

def test_dataloader_batch_shapes():
    loader = create_dataloader(
        root=GOPRO_ROOT, split="train", batch_size=4, crop_size=64, num_workers=0
    )
    batch = next(iter(loader))

    assert batch["input"].shape == (4, 3, 64, 64)
    assert batch["target"].shape == (4, 3, 64, 64)
    assert batch["task"][0] == "deblurring"
    assert len(batch["filename"]) == 4


def test_dataloader_from_config():
    with open("configs/srrt_base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # num_workers forced to 0 so the test does not spawn Windows workers.
    cfg["dataset"]["train"]["num_workers"] = 0
    cfg["dataset"]["val"]["num_workers"] = 0

    train_batch = next(iter(dataloader_from_config(cfg, "train")))
    assert train_batch["input"].shape == (2, 3, 128, 128)

    val_batch = next(iter(dataloader_from_config(cfg, "val")))
    assert val_batch["input"].shape == (1, 3, 256, 256)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
