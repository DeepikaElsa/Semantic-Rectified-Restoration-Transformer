import pytest
import torch

from models.modules.sra import SRA
from tests._utils import assert_no_quadratic_attention


def test_output_shape_preserved():
    model = SRA(dim=96, num_heads=4)
    x = torch.randn(2, 96, 32, 32)
    assert model(x).shape == x.shape


def test_is_exact_noop_at_initialization():
    """
    gamma is zero-initialized, so F_rect == F_R exactly at init. This keeps the
    SRA ablation honest (see module docstring).
    """
    model = SRA(dim=48, num_heads=4)
    x = torch.randn(1, 48, 16, 16)
    assert torch.allclose(model(x), x, atol=1e-7)


def test_becomes_active_once_gamma_is_nonzero():
    model = SRA(dim=48, num_heads=4)
    x = torch.randn(1, 48, 16, 16)

    with torch.no_grad():
        model.gamma.fill_(1.0)

    assert not torch.allclose(model(x), x, atol=1e-5)


def test_attention_map_is_a_valid_gate():
    model = SRA(dim=48, num_heads=4)
    _, maps = model(torch.randn(1, 48, 16, 16), return_maps=True)

    gate = maps["attention"]
    assert gate.shape == (1, 48, 16, 16)
    assert (gate >= 0).all() and (gate <= 1).all(), "sigmoid gate outside [0,1]"


def test_return_maps_shapes():
    model = SRA(dim=32, num_heads=2)
    out, maps = model(torch.randn(1, 32, 8, 8), return_maps=True)

    assert out.shape == (1, 32, 8, 8)
    assert maps["attention"].shape == (1, 32, 8, 8)
    assert maps["residual"].shape == (1, 32, 8, 8)


def test_rejects_indivisible_head_count():
    with pytest.raises(ValueError, match="divisible"):
        SRA(dim=30, num_heads=4)


def test_gradients_flow_including_gamma():
    model = SRA(dim=32, num_heads=2)
    with torch.no_grad():
        model.gamma.fill_(0.5)

    x = torch.randn(1, 32, 8, 8, requires_grad=True)
    model(x).mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert model.gamma.grad is not None and torch.isfinite(model.gamma.grad).all()


def test_expansion_is_configurable():
    model = SRA(dim=32, num_heads=2, expansion=2.0)
    assert model.res_in.out_channels == 64
    assert model(torch.randn(1, 32, 8, 8)).shape == (1, 32, 8, 8)


def test_uses_channel_attention_not_spatial():
    """
    SRA runs at FULL decoder resolution, so a spatial attention map here would
    be catastrophic. Verify the attention stays channel-sized.
    """
    model = SRA(dim=96, num_heads=4)
    with assert_no_quadratic_attention(max_allowed_dim=96 // 4):
        model(torch.randn(1, 96, 64, 64))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
