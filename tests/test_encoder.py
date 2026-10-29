import torch

from models.backbone.encoder import Downsample, Encoder, OverlapPatchEmbed
from tests._utils import assert_no_quadratic_attention, get_device, measure_resources


def test_patch_embed_shape():
    x = torch.randn(2, 3, 32, 32)
    embed = OverlapPatchEmbed(in_channels=3, dim=48)
    assert embed(x).shape == (2, 48, 32, 32)


def test_downsample_halves_spatial_doubles_channels():
    x = torch.randn(2, 48, 32, 32)
    down = Downsample(48)
    assert down(x).shape == (2, 96, 16, 16)


def test_encoder_shapes_small():
    # H=W=32 is divisible by 8 (three downsample-by-2 stages).
    x = torch.randn(1, 3, 32, 32)

    model = Encoder(dim=48, encoder_blocks=(1, 1, 1), heads=(1, 2, 4, 8))
    x1, x2, x3, x4 = model(x)

    assert x1.shape == (1, 48, 32, 32)
    assert x2.shape == (1, 96, 16, 16)
    assert x3.shape == (1, 192, 8, 8)
    assert x4.shape == (1, 384, 4, 4)


def test_encoder_config_driven_dim():
    x = torch.randn(1, 3, 16, 16)

    model = Encoder(dim=16, encoder_blocks=(1, 1, 1), heads=(1, 1, 1, 1))
    x1, x2, x3, x4 = model(x)

    assert x1.shape == (1, 16, 16, 16)
    assert x2.shape == (1, 32, 8, 8)
    assert x3.shape == (1, 64, 4, 4)
    assert x4.shape == (1, 128, 2, 2)


def test_encoder_rejects_wrong_length_config():
    for bad in [(1, 1), (1, 1, 1, 1)]:
        try:
            Encoder(dim=16, encoder_blocks=bad, heads=(1, 1, 1, 1))
        except ValueError:
            continue
        raise AssertionError(f"Encoder accepted invalid encoder_blocks={bad}")


def test_encoder_outputs_finite_and_gradients_flow():
    x = torch.randn(1, 3, 32, 32, requires_grad=True)

    model = Encoder(dim=16, encoder_blocks=(1, 1, 1), heads=(1, 1, 1, 1))
    outs = model(x)

    for t in outs:
        assert torch.isfinite(t).all()

    sum(t.sum() for t in outs).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_encoder_realistic_resolution():
    device = get_device()
    x = torch.randn(1, 3, 256, 256, device=device)

    # dim=48 matches configs/srrt_base.yaml defaults; this is the exact
    # scenario (256x256, dim=48, level-1 heads=1) that previously triggered
    # the ~34GB HW x HW allocation.
    model = Encoder(dim=48, encoder_blocks=(4, 6, 6), heads=(1, 2, 4, 8)).to(device)

    with assert_no_quadratic_attention(max_allowed_dim=384):
        with measure_resources(device) as stats, torch.no_grad():
            x1, x2, x3, x4 = model(x)

    assert x1.shape == (1, 48, 256, 256)
    assert x2.shape == (1, 96, 128, 128)
    assert x3.shape == (1, 192, 64, 64)
    assert x4.shape == (1, 384, 32, 32)

    print(f"[Encoder @ 256x256] time={stats['elapsed_s']:.3f}s "
          f"gpu_peak={stats['gpu_peak_mb']:.1f}MB cpu_delta={stats['cpu_delta_mb']:.1f}MB")


if __name__ == "__main__":
    test_patch_embed_shape()
    test_downsample_halves_spatial_doubles_channels()
    test_encoder_shapes_small()
    test_encoder_config_driven_dim()
    test_encoder_rejects_wrong_length_config()
    test_encoder_outputs_finite_and_gradients_flow()
    test_encoder_realistic_resolution()
    print("All encoder tests passed.")
