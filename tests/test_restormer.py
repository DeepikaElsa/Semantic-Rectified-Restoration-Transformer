import yaml
import torch

from models.backbone.restormer import Restormer
from tests._utils import assert_no_quadratic_attention, get_device, measure_resources


def _tiny_model():
    return Restormer(
        dim=16,
        encoder_blocks=(1, 1, 1, 1),
        decoder_blocks=(1, 1, 1),
        heads=(1, 1, 1, 1),
        refinement_blocks=1,
    )


def test_cuda_availability_reported():
    # Not an assertion of truth (CPU-only machines are valid) -- just verifies
    # the check runs and the chosen device is usable.
    available = torch.cuda.is_available()
    device = get_device()
    print(f"torch.cuda.is_available() = {available}, using device = {device}")
    assert torch.zeros(1, device=device).device.type == device.type


def test_restormer_shape_small_tensor():
    x = torch.randn(1, 3, 32, 32)
    assert _tiny_model()(x).shape == x.shape


def test_restormer_forward_features_exposes_intermediates():
    x = torch.randn(1, 3, 32, 32)
    feats = _tiny_model().forward_features(x)

    expected = {"skip1", "skip2", "skip3", "latent", "decoded", "refined", "residual", "output"}
    assert set(feats) == expected

    assert feats["skip1"].shape == (1, 16, 32, 32)
    assert feats["skip2"].shape == (1, 32, 16, 16)
    assert feats["skip3"].shape == (1, 64, 8, 8)
    assert feats["latent"].shape == (1, 128, 4, 4)
    assert feats["decoded"].shape == (1, 32, 32, 32)
    assert feats["refined"].shape == (1, 32, 32, 32)
    assert feats["residual"].shape == (1, 3, 32, 32)
    assert feats["output"].shape == (1, 3, 32, 32)

    for name, t in feats.items():
        assert torch.isfinite(t).all(), f"{name} contains non-finite values"


def test_restormer_global_residual_is_applied():
    """output must equal residual + input (global residual connection)."""
    x = torch.randn(1, 3, 32, 32)
    feats = _tiny_model().forward_features(x)
    assert torch.allclose(feats["output"], feats["residual"] + x, atol=1e-6)


def test_restormer_backward_and_gradients():
    model = _tiny_model()
    x = torch.randn(1, 3, 32, 32)

    loss = model(x).mean()
    loss.backward()

    n_with_grad = sum(
        1 for p in model.parameters()
        if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    )
    assert n_with_grad > 0, "no parameter received a finite non-zero gradient"


def test_restormer_builds_from_yaml_config():
    with open("configs/srrt_base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    model = Restormer.from_config(cfg)
    r = cfg["model"]["restormer"]

    assert model.dim == r["dim"]
    assert model.encoder.patch_embed.proj.out_channels == r["dim"]
    assert model.latent_dim == r["dim"] * 8


def test_restormer_rejects_wrong_encoder_blocks_length():
    try:
        Restormer(dim=16, encoder_blocks=(1, 1, 1))
    except ValueError:
        return
    raise AssertionError("Restormer accepted encoder_blocks of length 3")


def test_restormer_realistic_image_tensor():
    """
    Full pipeline at B=1, C=3, H=W=256 with the default architecture from
    configs/srrt_base.yaml. This is the configuration that previously caused a
    ~34GB allocation and a CPU RAM crash; it must now complete cheaply.
    """
    device = get_device()

    with open("configs/srrt_base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    model = Restormer.from_config(cfg).to(device).eval()
    x = torch.randn(1, 3, 256, 256, device=device)

    with assert_no_quadratic_attention(max_allowed_dim=384):
        with measure_resources(device) as stats, torch.no_grad():
            y = model(x)

    assert y.shape == x.shape
    n_params = sum(p.numel() for p in model.parameters())

    print(f"[Restormer @ 256x256] time={stats['elapsed_s']:.3f}s "
          f"gpu_peak={stats['gpu_peak_mb']:.1f}MB cpu_delta={stats['cpu_delta_mb']:.1f}MB "
          f"params={n_params:,}")

    if device.type == "cuda":
        assert stats["gpu_peak_mb"] < 6000


if __name__ == "__main__":
    test_cuda_availability_reported()
    test_restormer_shape_small_tensor()
    test_restormer_forward_features_exposes_intermediates()
    test_restormer_global_residual_is_applied()
    test_restormer_backward_and_gradients()
    test_restormer_builds_from_yaml_config()
    test_restormer_rejects_wrong_encoder_blocks_length()
    test_restormer_realistic_image_tensor()
    print("All restormer tests passed.")
