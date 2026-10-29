"""
Training sanity tests (spec section 21).

TEST 1: one-batch forward pass
TEST 2: one-batch backward pass
TEST 3: optimizer step actually changes the weights
TEST 4: the model can overfit a tiny fixed batch (loss decreases)

These must pass before any full training run is attempted. They use a small
Restormer so the suite stays fast; test_restormer.py covers the full-size
architecture separately.
"""

import copy

import torch
import yaml

from losses.combined_loss import CombinedLoss
from models.backbone.restormer import Restormer
from tests._utils import get_device
from train import build_optimizer, build_scheduler, compute_psnr


def _small_model():
    return Restormer(
        dim=16,
        encoder_blocks=(1, 1, 1, 1),
        decoder_blocks=(1, 1, 1),
        heads=(1, 1, 1, 1),
        refinement_blocks=1,
    )


def _batch(device, n=2, size=32):
    return (
        torch.rand(n, 3, size, size, device=device),
        torch.rand(n, 3, size, size, device=device),
    )


def test_1_forward_pass():
    device = get_device()
    model = _small_model().to(device)
    inp, _ = _batch(device)

    with torch.no_grad():
        out = model(inp)

    assert out.shape == inp.shape
    assert torch.isfinite(out).all()


def test_2_backward_pass():
    device = get_device()
    model = _small_model().to(device)
    loss_fn = CombinedLoss(l1_weight=1.0, perceptual_weight=0.0, use_perceptual=False)

    inp, tgt = _batch(device)
    loss, _ = loss_fn(model(inp), tgt)
    loss.backward()

    assert torch.isfinite(loss)

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients were produced"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient detected"
    assert sum(g.abs().sum() for g in grads) > 0, "all gradients are zero"


def test_3_optimizer_step_changes_weights():
    device = get_device()
    with open("configs/srrt_base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    model = _small_model().to(device)
    loss_fn = CombinedLoss(l1_weight=1.0, perceptual_weight=0.0, use_perceptual=False)
    optimizer = build_optimizer(cfg, model)

    before = copy.deepcopy(model.state_dict())

    inp, tgt = _batch(device)
    optimizer.zero_grad(set_to_none=True)
    loss, _ = loss_fn(model(inp), tgt)
    loss.backward()
    optimizer.step()

    after = model.state_dict()
    changed = sum(1 for k in before if not torch.equal(before[k], after[k]))
    assert changed > 0, "optimizer.step() did not change any parameter"


def test_3b_scheduler_advances_learning_rate():
    with open("configs/srrt_base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    model = _small_model()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    lr_before = optimizer.param_groups[0]["lr"]

    # optimizer.step() before scheduler.step() (PyTorch >=1.1 ordering).
    model(torch.randn(1, 3, 32, 32)).mean().backward()
    optimizer.step()
    scheduler.step()

    lr_after = optimizer.param_groups[0]["lr"]

    assert lr_before == cfg["optimizer"]["learning_rate"]
    assert lr_after < lr_before, "cosine schedule did not decrease the learning rate"


def test_4_overfits_tiny_real_batch():
    """
    The clearest end-to-end signal that the whole chain learns: on a tiny fixed
    subset of REAL GoPro data the loss must fall substantially and PSNR must
    improve on the model's own starting point.

    Real image pairs are used deliberately -- random-noise targets are
    incompressible and would test nothing about restoration.
    """
    device = get_device()
    torch.manual_seed(0)

    from data.datasets import create_dataloader

    loader = create_dataloader(
        root="data/Gopro", split="train", batch_size=2, crop_size=64,
        crop_mode="center", augment=False, shuffle=False, num_workers=0,
    )
    batch = next(iter(loader))
    inp = batch["input"].to(device)
    tgt = batch["target"].to(device)

    model = _small_model().to(device)
    loss_fn = CombinedLoss(l1_weight=1.0, perceptual_weight=0.0, use_perceptual=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with torch.no_grad():
        psnr_before = compute_psnr(model(inp).clamp(0, 1), tgt)

    losses = []
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = loss_fn(model(inp), tgt)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    first, last = losses[0], losses[-1]
    with torch.no_grad():
        psnr_after = compute_psnr(model(inp).clamp(0, 1), tgt)

    print(f"[overfit] loss {first:.4f} -> {last:.4f}, "
          f"PSNR {psnr_before:.2f}dB -> {psnr_after:.2f}dB")

    assert all(torch.isfinite(torch.tensor(x)) for x in losses)
    assert last < first * 0.5, f"loss did not halve while overfitting: {first:.4f} -> {last:.4f}"
    assert psnr_after > psnr_before, (
        f"PSNR did not improve while overfitting: {psnr_before:.2f} -> {psnr_after:.2f}dB"
    )


if __name__ == "__main__":
    test_1_forward_pass()
    test_2_backward_pass()
    test_3_optimizer_step_changes_weights()
    test_3b_scheduler_advances_learning_rate()
    test_4_overfits_tiny_real_batch()
    print("All training sanity tests passed.")
