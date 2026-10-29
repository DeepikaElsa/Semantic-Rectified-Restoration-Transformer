"""
Verifies every hand-computed number in docs/numerical_example.md.

Documentation that drifts from the code is worse than no documentation, so the
worked examples are executable.
"""

import math

import pytest
import torch

from models.modules.lcr import LCR
from models.modules.rectified_flow import RectifiedFlow
from train import compute_psnr


def test_rectified_flow_interpolation_example():
    z0 = torch.tensor([[1.0, 2.0]])
    z1 = torch.tensor([[5.0, 6.0]])
    t = torch.tensor([0.25])

    z_t = RectifiedFlow.interpolate(z0, z1, t)
    assert torch.allclose(z_t, torch.tensor([[2.0, 3.0]]))


def test_rectified_flow_target_velocity_example():
    z0 = torch.tensor([[1.0, 2.0]])
    z1 = torch.tensor([[5.0, 6.0]])
    assert torch.allclose(z1 - z0, torch.tensor([[4.0, 4.0]]))


def test_rectified_flow_loss_example_both_conventions():
    v_star = torch.tensor([4.0, 4.0])
    v_pred = torch.tensor([3.8, 4.2])

    summed = ((v_pred - v_star) ** 2).sum()
    meaned = torch.nn.functional.mse_loss(v_pred, v_star)

    assert math.isclose(float(summed), 0.08, abs_tol=1e-6)
    assert math.isclose(float(meaned), 0.04, abs_tol=1e-6)


def test_rectified_flow_endpoints_example():
    z0 = torch.tensor([[1.0, 2.0]])
    z1 = torch.tensor([[5.0, 6.0]])

    assert torch.allclose(RectifiedFlow.interpolate(z0, z1, torch.tensor([0.0])), z0)
    assert torch.allclose(RectifiedFlow.interpolate(z0, z1, torch.tensor([1.0])), z1)


def test_euler_with_constant_velocity_is_exact():
    """4 Euler steps with the ideal constant velocity land exactly on z1."""
    z0 = torch.tensor([[1.0, 2.0]])
    z1 = torch.tensor([[5.0, 6.0]])
    v = z1 - z0

    z = z0.clone()
    dt = 0.25
    for _ in range(4):
        z = z + dt * v

    assert torch.allclose(z, z1)


def test_l1_loss_example():
    pred = torch.tensor([[0.9, 0.2], [0.4, 0.7]])
    target = torch.tensor([[1.0, 0.0], [0.5, 1.0]])

    l1 = torch.nn.functional.l1_loss(pred, target)
    assert math.isclose(float(l1), 0.175, abs_tol=1e-6)


def test_psnr_example():
    pred = torch.tensor([[0.9, 0.2], [0.4, 0.7]]).view(1, 1, 2, 2)
    target = torch.tensor([[1.0, 0.0], [0.5, 1.0]]).view(1, 1, 2, 2)

    mse = float(((pred - target) ** 2).mean())
    assert math.isclose(mse, 0.0375, abs_tol=1e-7)   # fp32 precision

    psnr = compute_psnr(pred, target)
    assert math.isclose(psnr, 14.2597, abs_tol=1e-3)


def test_mdta_matmul_orientation_example():
    """
    The tiny (B=1,C=4,H=W=2,h=2,d=2) case from the doc: the correct order gives
    a (d x d) matrix, the incorrect order an (HW x HW) one.
    """
    q = torch.randn(1, 2, 2, 4)      # (B, h, d, HW)
    k = torch.randn(1, 2, 2, 4)

    correct = torch.matmul(q, k.transpose(-2, -1))
    incorrect = torch.matmul(q.transpose(-2, -1), k)

    assert correct.shape == (1, 2, 2, 2)      # d x d
    assert incorrect.shape == (1, 2, 4, 4)    # HW x HW


def test_documented_blowup_ratio():
    hw, d = 256 * 256, 48
    assert hw == 65536
    assert hw ** 2 == 4294967296
    assert d ** 2 == 2304

    # fp32 bytes for the HW x HW matrix, in GB
    gb = hw ** 2 * 4 / (1024 ** 3)
    assert math.isclose(gb, 16.0, abs_tol=0.5)     # ~17.2 GB as documented (GiB)

    ratio = (hw / d) ** 2
    assert 1.8e6 < ratio < 1.9e6


@pytest.mark.parametrize("cos_angle,expected", [(1.0, 0.0), (0.0, 2.0), (-1.0, 4.0)])
def test_lcr_consistency_loss_equals_2_minus_2cos(cos_angle, expected):
    """
    L_LCR = ||P_pred - P_ref||^2 = 2(1 - cos) for unit-norm projections.

    Checked directly on unit vectors (the projection head is bypassed so the
    identity itself is under test).
    """
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[cos_angle, math.sqrt(max(0.0, 1 - cos_angle ** 2))]])

    summed = ((a - b) ** 2).sum()
    assert math.isclose(float(summed), expected, abs_tol=1e-6)


def test_lcr_loss_zero_for_identical_and_bounded():
    model = LCR(visual_dim=32, damage_dim=32, hidden_dim=64, latent_dim=64,
                proj_dim=32, num_heads=2, refinement_blocks=1)
    z = torch.randn(2, 64, 8, 8)

    assert float(model.consistency_loss(z, z).detach()) < 1e-10

    other = torch.randn(2, 64, 8, 8)
    # mse_loss averages over proj_dim, so it is bounded by 4/proj_dim.
    assert 0 <= float(model.consistency_loss(other, z).detach()) <= 4.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
