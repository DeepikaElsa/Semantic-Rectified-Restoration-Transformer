import pytest
import torch

from models.modules.diffae import DiffAE


def test_latent_shapes():
    model = DiffAE(dim=32, latent_dim=256)
    out = model(torch.randn(1, 3, 128, 128))

    assert out["spatial"].shape == (1, 256, 16, 16)
    assert out["semantic"].shape == (1, 256)
    assert "recon" not in out


def test_spatial_latent_is_at_stride_8():
    model = DiffAE(dim=8, latent_dim=32)
    out = model(torch.randn(1, 3, 64, 64))
    assert out["spatial"].shape[-2:] == (64 // model.out_stride, 64 // model.out_stride)


def test_configurable_latent_dim():
    out = DiffAE(dim=8, latent_dim=48)(torch.randn(2, 3, 32, 32))
    assert out["spatial"].shape == (2, 48, 4, 4)
    assert out["semantic"].shape == (2, 48)


def test_optional_reconstruction_head():
    model = DiffAE(dim=8, latent_dim=32, with_decoder=True)
    out = model(torch.randn(1, 3, 64, 64), return_reconstruction=True)
    assert out["recon"].shape == (1, 3, 64, 64)


def test_reconstruction_requires_decoder():
    model = DiffAE(dim=8, latent_dim=32, with_decoder=False)
    with pytest.raises(RuntimeError, match="with_decoder=False"):
        model(torch.randn(1, 3, 32, 32), return_reconstruction=True)


def test_outputs_finite_and_gradients_flow():
    model = DiffAE(dim=8, latent_dim=32)
    x = torch.randn(1, 3, 32, 32, requires_grad=True)

    out = model(x)
    assert torch.isfinite(out["spatial"]).all()
    assert torch.isfinite(out["semantic"]).all()

    (out["spatial"].mean() + out["semantic"].mean()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_different_inputs_give_different_latents():
    """Sanity check that the encoder is input-dependent, not collapsed."""
    torch.manual_seed(0)
    model = DiffAE(dim=8, latent_dim=32).eval()

    with torch.no_grad():
        a = model(torch.rand(1, 3, 32, 32))["semantic"]
        b = model(torch.rand(1, 3, 32, 32))["semantic"]

    assert not torch.allclose(a, b, atol=1e-4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
