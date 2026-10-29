import pytest
import torch
import yaml

from losses.combined_loss import CombinedLoss, VGGPerceptualLoss, build_perceptual_loss
from tests._utils import get_device, measure_resources


def test_l1_only_matches_manual_l1():
    pred = torch.rand(2, 3, 32, 32)
    target = torch.rand(2, 3, 32, 32)

    loss_fn = CombinedLoss(l1_weight=1.0, perceptual_weight=0.0, use_perceptual=False)
    total, components = loss_fn(pred, target)

    assert torch.allclose(total, torch.nn.functional.l1_loss(pred, target), atol=1e-6)
    assert "perceptual" not in components


def test_identical_images_give_zero_loss():
    x = torch.rand(1, 3, 32, 32)
    _, components = CombinedLoss(l1_weight=1.0, perceptual_weight=0.1)(x, x)

    assert components["l1"].item() < 1e-6
    assert components["perceptual"].item() < 1e-4
    assert components["total"].item() < 1e-4


def test_weights_are_applied():
    pred = torch.rand(1, 3, 16, 16)
    target = torch.rand(1, 3, 16, 16)

    loss_fn = CombinedLoss(l1_weight=2.5, perceptual_weight=0.0, use_perceptual=False)
    total, components = loss_fn(pred, target)

    assert torch.allclose(total, 2.5 * components["l1"], atol=1e-6)


def test_optional_flow_and_lcr_terms():
    pred = torch.rand(1, 3, 16, 16)
    target = torch.rand(1, 3, 16, 16)

    loss_fn = CombinedLoss(
        l1_weight=1.0, perceptual_weight=0.0, flow_weight=0.5,
        consistency_weight=0.25, use_perceptual=False,
    )

    # Baseline: neither term supplied -> absent from components.
    _, base = loss_fn(pred, target)
    assert "flow" not in base and "lcr" not in base

    flow = torch.tensor(2.0)
    lcr = torch.tensor(4.0)
    total, components = loss_fn(pred, target, flow_loss=flow, lcr_loss=lcr)

    assert components["flow"] is flow and components["lcr"] is lcr
    expected = base["l1"] + 0.5 * flow + 0.25 * lcr
    assert torch.allclose(total, expected, atol=1e-6)


def test_gradients_flow_to_prediction():
    pred = torch.rand(1, 3, 32, 32, requires_grad=True)
    target = torch.rand(1, 3, 32, 32)

    total, _ = CombinedLoss(l1_weight=1.0, perceptual_weight=0.1)(pred, target)
    total.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_perceptual_network_stays_frozen_and_in_eval():
    loss_fn = CombinedLoss(l1_weight=1.0, perceptual_weight=0.1)
    loss_fn.train()  # training loop calls this; VGG must stay in eval mode

    assert isinstance(loss_fn.perceptual, VGGPerceptualLoss)
    assert not loss_fn.perceptual.vgg.training
    assert all(not p.requires_grad for p in loss_fn.perceptual.vgg.parameters())


def test_build_perceptual_loss_rejects_unknown_type():
    with pytest.raises(ValueError):
        build_perceptual_loss("resnet50")


def test_builds_from_yaml_config():
    with open("configs/srrt_base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    loss_fn = CombinedLoss.from_config(cfg)

    assert loss_fn.l1_weight == cfg["loss"]["l1_weight"]
    assert loss_fn.perceptual_weight == cfg["loss"]["perceptual_weight"]
    assert loss_fn.flow_weight == cfg["loss"]["flow_weight"]
    assert loss_fn.consistency_weight == cfg["loss"]["consistency_weight"]
    # Config must default to VGG19 so LPIPS stays an independent eval metric.
    assert cfg["loss"]["perceptual_type"] == "vgg19"


def test_realistic_resolution():
    device = get_device()
    pred = torch.rand(1, 3, 256, 256, device=device)
    target = torch.rand(1, 3, 256, 256, device=device)

    loss_fn = CombinedLoss(l1_weight=1.0, perceptual_weight=0.1).to(device)

    with measure_resources(device) as stats:
        total, _ = loss_fn(pred, target)

    assert torch.isfinite(total)
    print(f"[CombinedLoss @ 256x256] time={stats['elapsed_s']:.3f}s "
          f"gpu_peak={stats['gpu_peak_mb']:.1f}MB cpu_delta={stats['cpu_delta_mb']:.1f}MB")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
