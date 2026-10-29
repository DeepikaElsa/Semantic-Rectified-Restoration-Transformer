"""End-to-end tests for the integrated SRRT model and every ablation stage."""

import pytest
import torch
import yaml

from models.backbone.restormer import Restormer
from models.srrt import SRRT
from tests._utils import assert_no_quadratic_attention, get_device
from utils.ablation import ABLATIONS, apply_ablation


def _tiny_cfg():
    """Small architecture so the suite stays fast; shapes/logic are unchanged."""
    with open("configs/srrt_base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    r = cfg["model"]["restormer"]
    r["dim"] = 16
    r["encoder_blocks"] = [1, 1, 1, 1]
    r["decoder_blocks"] = [1, 1, 1]
    r["num_heads"] = [1, 1, 1, 1]
    r["refinement_blocks"] = 1
    cfg["model"]["latent_dim"] = 16 * 8

    m = cfg["model"]["modules"]
    m["seecoder"].update({"embed_dim": 32, "dim": 8, "num_heads": 1, "blocks_per_stage": 1})
    m["diffae"].update({"latent_dim": 32, "dim": 8})
    m["lcr"].update({"hidden_dim": 64, "proj_dim": 32, "num_heads": 2, "refinement_blocks": 1})
    m["rectified_flow"].update({"hidden_dim": 32, "time_embed_dim": 32, "num_blocks": 1, "steps": 4})
    m["sra"].update({"num_heads": 2})
    return cfg


def test_baseline_srrt_is_numerically_identical_to_restormer():
    """
    With every module disabled, SRRT must reproduce Restormer EXACTLY.

    This is what makes the ablation trustworthy: any measured difference at
    stages B..F is attributable to the added module, not to incidental changes
    in the backbone path.
    """
    cfg = apply_ablation(_tiny_cfg(), "baseline_restormer")
    torch.manual_seed(0)

    srrt = SRRT.from_config(cfg).eval()
    restormer = Restormer.from_config(cfg).eval()
    restormer.load_state_dict(srrt.backbone.state_dict())

    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(srrt(x)["output"], restormer(x), atol=1e-6)


@pytest.mark.parametrize("name", list(ABLATIONS))
def test_every_ablation_runs_forward_and_backward(name):
    cfg = apply_ablation(_tiny_cfg(), name)
    model = SRRT.from_config(cfg)

    x = torch.randn(1, 3, 32, 32)
    tgt = torch.randn(1, 3, 32, 32)

    with assert_no_quadratic_attention(max_allowed_dim=128):
        out = model(x, target=tgt)

    assert out["output"].shape == x.shape
    assert torch.isfinite(out["output"]).all()

    total = out["output"].mean()
    if "flow_loss" in out:
        assert torch.isfinite(out["flow_loss"])
        total = total + out["flow_loss"]
    if "lcr_loss" in out:
        assert torch.isfinite(out["lcr_loss"])
        total = total + out["lcr_loss"]

    total.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, name + ": no gradients"
    assert all(torch.isfinite(g).all() for g in grads), name + ": non-finite gradient"


def test_loss_terms_present_only_for_enabled_modules():
    x = torch.randn(1, 3, 32, 32)
    tgt = torch.randn(1, 3, 32, 32)

    base = SRRT.from_config(apply_ablation(_tiny_cfg(), "baseline_restormer"))(x, target=tgt)
    assert "flow_loss" not in base and "lcr_loss" not in base

    lcr = SRRT.from_config(apply_ablation(_tiny_cfg(), "seecoder_diffae_lcr"))(x, target=tgt)
    assert "lcr_loss" in lcr and "flow_loss" not in lcr

    flow = SRRT.from_config(apply_ablation(_tiny_cfg(), "seecoder_diffae_lcr_flow"))(x, target=tgt)
    assert "flow_loss" in flow and "lcr_loss" in flow


def test_inference_without_target_uses_ode_and_omits_losses():
    model = SRRT.from_config(apply_ablation(_tiny_cfg(), "full_srrt")).eval()

    with torch.no_grad():
        out = model(torch.randn(1, 3, 32, 32))

    assert "flow_loss" not in out and "lcr_loss" not in out
    assert "latent_flow" in out
    assert torch.isfinite(out["output"]).all()


def test_invalid_module_combinations_rejected():
    cfg = _tiny_cfg()
    cfg["model"]["modules"]["lcr"]["enabled"] = True          # without seecoder/diffae
    with pytest.raises(ValueError, match="requires BOTH"):
        SRRT.from_config(cfg)

    cfg = apply_ablation(_tiny_cfg(), "seecoder_diffae")
    cfg["model"]["modules"]["rectified_flow"]["enabled"] = True  # without lcr
    with pytest.raises(ValueError, match="requires lcr"):
        SRRT.from_config(cfg)


def test_restore_convenience_api():
    model = SRRT.from_config(apply_ablation(_tiny_cfg(), "full_srrt")).eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert model.restore(x).shape == x.shape


def test_return_features_exposes_intermediates():
    model = SRRT.from_config(apply_ablation(_tiny_cfg(), "full_srrt")).eval()

    with torch.no_grad():
        out = model(torch.randn(1, 3, 32, 32), return_features=True)

    for key in ("visual_features", "damage_features", "latent_initial",
                "latent_flow", "sra_attention", "sra_residual", "latent", "residual"):
        assert key in out, "missing " + key


def test_module_summary_counts_parameters():
    model = SRRT.from_config(apply_ablation(_tiny_cfg(), "full_srrt"))
    summary = model.module_summary()

    assert all(summary[k]["enabled"] for k in ("seecoder", "diffae", "lcr", "rectified_flow", "sra"))
    assert summary["total"] > summary["backbone"]["params"]


def test_full_srrt_at_realistic_resolution():
    """Full-size SRRT at 256x256 must not build an HW x HW attention matrix."""
    device = get_device()

    with open("configs/srrt_base.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_ablation(cfg, "full_srrt")

    model = SRRT.from_config(cfg).to(device).eval()
    x = torch.randn(1, 3, 256, 256, device=device)

    with assert_no_quadratic_attention(max_allowed_dim=384):
        with torch.no_grad():
            out = model(x)

    assert out["output"].shape == x.shape
    assert torch.isfinite(out["output"]).all()
    print("[full SRRT @ 256] params:", format(model.module_summary()["total"], ","))


def _grad_norm(model, loss, prefix):
    """L2 norm of the gradient `loss` deposits on parameters under `prefix`."""
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    total = 0.0
    for name, p in model.named_parameters():
        if name.startswith(prefix) and p.grad is not None:
            total += float(p.grad.norm()) ** 2
    return total ** 0.5


def test_flow_target_is_detached_from_the_restoration_encoder():
    """
    The rectified-flow regression target must be a CONSTANT, not a tensor the
    optimizer can move.

    z1 = encode_latent(target) comes from the shared restoration encoder. Left
    in the graph, `flow_loss = ||v(z_t,t) - (z1 - z0)||^2` can be minimized by
    dragging z1 toward z0 -- that is, by degrading the encoder the restoration
    path depends on, which is a strictly easier descent direction than learning
    the velocity field.

    Regression: with z1 attached, flow_loss deposited MORE gradient on
    backbone.encoder (4.23e-02) than on the velocity network (1.97e-02), and
    ablations E/F collapsed ~10 dB at epoch 6 while flow_loss fell 31x.
    """
    device = get_device()
    cfg = apply_ablation(_tiny_cfg(), "seecoder_diffae_lcr_flow")

    torch.manual_seed(0)
    model = SRRT.from_config(cfg).to(device).train()
    y = torch.randn(2, 3, 64, 64, device=device)
    target = torch.randn(2, 3, 64, 64, device=device)

    out = model(y, target=target)
    flow_loss = out["flow_loss"]
    assert flow_loss is not None, "flow_loss missing with rectified_flow enabled"

    # (1)+(2) no gradient may reach the encoder or bottleneck through the target
    enc = _grad_norm(model, flow_loss, "backbone.encoder")
    bot = _grad_norm(model, flow_loss, "backbone.bottleneck")
    assert enc == 0.0, (
        f"flow_loss leaks {enc:.3e} gradient into backbone.encoder; the clean-target "
        "latent z1 is not detached and the flow objective can corrupt the encoder"
    )
    assert bot == 0.0, f"flow_loss leaks {bot:.3e} gradient into backbone.bottleneck"

    # (3) the velocity network must still be trained by this loss
    vel = _grad_norm(model, flow_loss, "flow")
    assert vel > 0.0, "flow_loss no longer trains the velocity network"

    # (4) z0 stays differentiable: the modules that produce it must still learn
    for mod in ("seecoder", "diffae", "lcr"):
        g = _grad_norm(model, flow_loss, mod)
        assert g > 0.0, f"flow_loss no longer reaches '{mod}' (z0 path broken)"


def test_lcr_reference_remains_detached():
    """(6) D's protective behaviour is unchanged: lcr_loss never touches the encoder."""
    device = get_device()
    cfg = apply_ablation(_tiny_cfg(), "seecoder_diffae_lcr")

    torch.manual_seed(0)
    model = SRRT.from_config(cfg).to(device).train()
    y = torch.randn(2, 3, 64, 64, device=device)
    target = torch.randn(2, 3, 64, 64, device=device)

    out = model(y, target=target)
    assert out.get("lcr_loss") is not None
    assert _grad_norm(model, out["lcr_loss"], "backbone.encoder") == 0.0
    assert _grad_norm(model, out["lcr_loss"], "lcr") > 0.0


@pytest.mark.parametrize("ablation", [a for a in sorted(ABLATIONS) if a != "baseline_restormer"])
def test_enabled_modules_are_actually_connected(ablation):
    """
    Every module an ablation enables must be able to reach the output.

    The zero-gate check cannot catch this on its own: a module wired through a
    zero-initialized gate and a module wired to nothing at all both produce
    output identical to Restormer at init. They differ in what happens *after*
    the first optimizer step -- a gated module starts receiving gradient, a
    disconnected one never does.

    Regression: `seecoder` built its bottleneck injection only when LCR was
    enabled, so `--ablation seecoder` trained 335,380 parameters that had no
    path to the loss and never received a gradient.
    """
    device = get_device()
    cfg = apply_ablation(_tiny_cfg(), ablation)

    torch.manual_seed(0)
    model = SRRT.from_config(cfg).to(device).train()
    x = torch.randn(1, 3, 64, 64, device=device)
    target = torch.randn(1, 3, 64, 64, device=device)

    enabled = [name for name, spec in model.module_summary().items()
               if isinstance(spec, dict) and spec.get("enabled") and name != "backbone"]
    assert enabled, f"{ablation} enables no modules"

    opt = torch.optim.SGD(model.parameters(), lr=1.0)

    def objective():
        """The training objective, not just the pixel term.

        With Rectified Flow on, the backbone is fed the supervised endpoint z1
        during training, so the SeeCoder -> LCR -> z0 path reaches the loss
        through `flow_loss` / `lcr_loss` rather than through `output`. Scoring
        only the pixel term would report those modules as disconnected when
        train.py does in fact train them.
        """
        out = model(x, target=target)
        total = out["output"].sub(target).pow(2).mean()
        for key in ("flow_loss", "lcr_loss"):
            if out.get(key) is not None:
                total = total + out[key]
        return total

    # Two steps: the first moves the gate off zero, the second lets gradient
    # reach the gated module itself.
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        objective().backward()
        opt.step()

    opt.zero_grad(set_to_none=True)
    objective().backward()

    # module_summary() reports "rectified_flow"; the attribute is `self.flow`.
    attr_of = {"rectified_flow": "flow"}

    for name in enabled:
        prefix = attr_of.get(name, name)
        params = [p for n, p in model.named_parameters() if n.startswith(prefix)]
        assert params, f"{ablation}: module {name} has no parameters"
        live = [p for p in params if p.grad is not None and float(p.grad.abs().sum()) > 0]
        assert live, (
            f"{ablation}: module '{name}' received no gradient from the loss. "
            "It is enabled but has no path to the output."
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
