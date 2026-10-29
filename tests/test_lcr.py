"""Unit tests for LCR -- the primary proposed novelty."""

import pytest
import torch

from models.modules.lcr import LCR
from tests._utils import assert_no_quadratic_attention


def _lcr(**kw):
    defaults = dict(visual_dim=256, damage_dim=256, hidden_dim=384,
                    latent_dim=384, proj_dim=384, num_heads=8, refinement_blocks=2)
    defaults.update(kw)
    return LCR(**defaults)


def test_fusion_output_shape():
    out = _lcr()(torch.randn(1, 256, 16, 16), torch.randn(1, 256, 16, 16))
    assert out["z0"].shape == (1, 384, 16, 16)
    assert out["fused"].shape == (1, 384, 16, 16)


def test_mismatched_spatial_sizes_are_resized():
    """F_d may arrive on a different grid than F_v; LCR must reconcile them."""
    out = _lcr()(torch.randn(1, 256, 16, 16), torch.randn(1, 256, 8, 8))
    assert out["z0"].shape == (1, 384, 16, 16)


def test_configurable_dims():
    model = _lcr(visual_dim=64, damage_dim=32, hidden_dim=64,
                 latent_dim=128, proj_dim=32, num_heads=2, refinement_blocks=1)
    out = model(torch.randn(2, 64, 8, 8), torch.randn(2, 32, 8, 8))
    assert out["z0"].shape == (2, 128, 8, 8)


def test_rejects_indivisible_head_count():
    with pytest.raises(ValueError, match="divisible"):
        _lcr(hidden_dim=100, num_heads=8)


def test_projection_is_l2_normalized():
    model = _lcr()
    p = model.project_latent(torch.randn(4, 384, 16, 16))
    assert p.shape == (4, 384)
    assert torch.allclose(p.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_consistency_loss_is_zero_for_identical_latents():
    model = _lcr()
    z = torch.randn(2, 384, 16, 16)
    assert model.consistency_loss(z, z).item() < 1e-10


def test_consistency_loss_positive_for_different_latents():
    model = _lcr()
    loss = model.consistency_loss(torch.randn(2, 384, 16, 16), torch.randn(2, 384, 16, 16))
    assert loss.item() > 0 and torch.isfinite(loss)


def test_consistency_loss_cannot_be_gamed_by_scaling():
    """
    L2-normalizing the projections means scaling a latent down cannot reduce
    the loss -- guarding the standard collapse failure mode.
    """
    model = _lcr().eval()
    z_pred = torch.randn(2, 384, 16, 16)
    z_ref = torch.randn(2, 384, 16, 16)

    with torch.no_grad():
        base = model.consistency_loss(z_pred, z_ref)
        shrunk = model.consistency_loss(z_pred * 0.01, z_ref)

    # Shrinking does change the projection (the head is non-linear) but must not
    # drive the loss toward zero the way an unnormalized distance would.
    assert shrunk.item() > base.item() * 0.1


def test_reference_latent_is_detached():
    """z_ref must not receive gradient: the clean latent is a fixed target."""
    model = _lcr()
    z_pred = torch.randn(1, 384, 8, 8, requires_grad=True)
    z_ref = torch.randn(1, 384, 8, 8, requires_grad=True)

    model.consistency_loss(z_pred, z_ref).backward()

    assert z_pred.grad is not None
    assert z_ref.grad is None


def test_gradients_flow_through_fusion():
    model = _lcr(hidden_dim=64, latent_dim=64, proj_dim=32, num_heads=2, refinement_blocks=1,
                 visual_dim=32, damage_dim=32)
    v = torch.randn(1, 32, 8, 8, requires_grad=True)
    d = torch.randn(1, 32, 8, 8, requires_grad=True)

    model(v, d)["z0"].mean().backward()

    assert v.grad is not None and torch.isfinite(v.grad).all()
    assert d.grad is not None and torch.isfinite(d.grad).all()


def test_uses_channel_attention_not_spatial():
    model = _lcr()
    with assert_no_quadratic_attention(max_allowed_dim=384 // 8):
        model(torch.randn(1, 256, 32, 32), torch.randn(1, 256, 32, 32))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
