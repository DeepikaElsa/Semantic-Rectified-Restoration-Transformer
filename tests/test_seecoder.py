import torch

from models.modules.seecoder import SeeCoder


def test_multiscale_shapes():
    model = SeeCoder(dim=48, embed_dim=256)
    out = model(torch.randn(1, 3, 128, 128))

    f1, f2, f3 = out["features"]
    assert f1.shape == (1, 48, 64, 64)
    assert f2.shape == (1, 96, 32, 32)
    assert f3.shape == (1, 192, 16, 16)
    assert out["context"].shape == (1, 256, 16, 16)
    assert out["global_vec"].shape == (1, 256)


def test_context_is_at_stride_8():
    model = SeeCoder(dim=16, embed_dim=32)
    out = model(torch.randn(1, 3, 64, 64))
    assert out["context"].shape[-2:] == (64 // model.out_stride, 64 // model.out_stride)


def test_configurable_dims():
    model = SeeCoder(dim=8, embed_dim=24, blocks_per_stage=1, num_heads=1)
    out = model(torch.randn(2, 3, 32, 32))
    assert out["context"].shape == (2, 24, 4, 4)


def test_outputs_finite_and_gradients_flow():
    model = SeeCoder(dim=16, embed_dim=32, blocks_per_stage=1, num_heads=1)
    x = torch.randn(1, 3, 32, 32, requires_grad=True)

    out = model(x)
    assert torch.isfinite(out["context"]).all()
    assert all(torch.isfinite(f).all() for f in out["features"])

    out["context"].mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_is_not_a_restormer_copy():
    """
    SeeCoder must stay lightweight relative to the backbone -- if it were a
    second Restormer, ablation gains would be uninterpretable (see module
    docstring).
    """
    from models.backbone.restormer import Restormer

    see = sum(p.numel() for p in SeeCoder(dim=48, embed_dim=256).parameters())
    backbone = sum(p.numel() for p in Restormer().parameters())
    assert see < backbone * 0.1, f"SeeCoder has {see:,} params vs backbone {backbone:,}"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
