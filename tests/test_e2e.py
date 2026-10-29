import torch
import yaml

from data.datasets import dataloader_from_config
from models.backbone.restormer import Restormer
from tests._utils import assert_no_quadratic_attention, get_device


def _config():
    with open("configs/srrt_base.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"]["train"]["num_workers"] = 0
    cfg["dataset"]["val"]["num_workers"] = 0
    return cfg


def test_gopro_batch_through_restormer():
    """Real GoPro data -> real model, at the configured training crop size."""
    device = get_device()
    cfg = _config()

    batch = next(iter(dataloader_from_config(cfg, "train")))
    model = Restormer.from_config(cfg).to(device).eval()

    inp = batch["input"].to(device)

    with assert_no_quadratic_attention(max_allowed_dim=384):
        with torch.no_grad():
            output = model(inp)

    assert output.shape == inp.shape
    assert output.shape == batch["target"].shape
    assert torch.isfinite(output).all()
    assert batch["task"][0] == "deblurring"


def test_gopro_val_batch_at_full_crop():
    """Validation path runs at the larger 256 crop without exploding memory."""
    device = get_device()
    cfg = _config()

    batch = next(iter(dataloader_from_config(cfg, "val")))
    model = Restormer.from_config(cfg).to(device).eval()

    inp = batch["input"].to(device)
    assert inp.shape[-2:] == (256, 256)

    with assert_no_quadratic_attention(max_allowed_dim=384):
        with torch.no_grad():
            output = model(inp)

    assert output.shape == inp.shape
    assert torch.isfinite(output).all()


if __name__ == "__main__":
    test_gopro_batch_through_restormer()
    test_gopro_val_batch_at_full_crop()
    print("End-to-end GoPro -> Restormer tests passed.")
