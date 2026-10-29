import torch

from models.backbone.bottleneck import Bottleneck
from models.backbone.decoder import Decoder, Upsample
from models.backbone.encoder import Encoder
from tests._utils import assert_no_quadratic_attention, get_device, measure_resources


def test_upsample_doubles_spatial_halves_channels():
    x = torch.randn(2, 96, 16, 16)
    up = Upsample(96)
    assert up(x).shape == (2, 48, 32, 32)


def test_bottleneck_preserves_shape():
    x = torch.randn(1, 384, 4, 4)
    bottleneck = Bottleneck(dim=48, num_blocks=1, num_heads=8)
    assert bottleneck.latent_dim == 384
    assert bottleneck(x).shape == x.shape


def test_encoder_bottleneck_decoder_shapes_small():
    x = torch.randn(1, 3, 32, 32)

    encoder = Encoder(dim=48, encoder_blocks=(1, 1, 1), heads=(1, 2, 4, 8))
    bottleneck = Bottleneck(dim=48, num_blocks=1, num_heads=8)
    decoder = Decoder(dim=48, decoder_blocks=(1, 1, 1), heads=(1, 2, 4, 8))

    x1, x2, x3, x4 = encoder(x)
    latent = bottleneck(x4)
    out = decoder(latent, x1, x2, x3)

    # Decoder level 1 keeps the concatenated dim*2 width (no reduce conv).
    assert out.shape == (1, 96, 32, 32)


def test_decoder_rejects_wrong_length_config():
    try:
        Decoder(dim=16, decoder_blocks=(1, 1), heads=(1, 1, 1, 1))
    except ValueError:
        return
    raise AssertionError("Decoder accepted invalid decoder_blocks length")


def test_decoder_gradients_flow():
    encoder = Encoder(dim=16, encoder_blocks=(1, 1, 1), heads=(1, 1, 1, 1))
    bottleneck = Bottleneck(dim=16, num_blocks=1, num_heads=1)
    decoder = Decoder(dim=16, decoder_blocks=(1, 1, 1), heads=(1, 1, 1, 1))

    x = torch.randn(1, 3, 32, 32, requires_grad=True)

    x1, x2, x3, x4 = encoder(x)
    out = decoder(bottleneck(x4), x1, x2, x3)

    assert torch.isfinite(out).all()

    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_encoder_decoder_realistic_resolution():
    device = get_device()
    x = torch.randn(1, 3, 256, 256, device=device)

    encoder = Encoder(dim=48, encoder_blocks=(4, 6, 6), heads=(1, 2, 4, 8)).to(device)
    bottleneck = Bottleneck(dim=48, num_blocks=8, num_heads=8).to(device)
    decoder = Decoder(dim=48, decoder_blocks=(6, 6, 4), heads=(1, 2, 4, 8)).to(device)

    with assert_no_quadratic_attention(max_allowed_dim=384):
        with measure_resources(device) as stats, torch.no_grad():
            x1, x2, x3, x4 = encoder(x)
            out = decoder(bottleneck(x4), x1, x2, x3)

    assert out.shape == (1, 96, 256, 256)

    print(f"[Encoder+Bottleneck+Decoder @ 256x256] time={stats['elapsed_s']:.3f}s "
          f"gpu_peak={stats['gpu_peak_mb']:.1f}MB cpu_delta={stats['cpu_delta_mb']:.1f}MB")


if __name__ == "__main__":
    test_upsample_doubles_spatial_halves_channels()
    test_bottleneck_preserves_shape()
    test_encoder_bottleneck_decoder_shapes_small()
    test_decoder_rejects_wrong_length_config()
    test_decoder_gradients_flow()
    test_encoder_decoder_realistic_resolution()
    print("All decoder tests passed.")
