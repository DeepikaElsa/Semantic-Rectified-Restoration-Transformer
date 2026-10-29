import pytest
import torch

from models.modules.rectified_flow import RectifiedFlow, timestep_embedding


def _rf(**kw):
    defaults = dict(latent_dim=64, hidden_dim=32, time_embed_dim=32, num_blocks=1, steps=10)
    defaults.update(kw)
    return RectifiedFlow(**defaults)


def test_timestep_embedding_shape_and_finiteness():
    emb = timestep_embedding(torch.rand(4), 32)
    assert emb.shape == (4, 32)
    assert torch.isfinite(emb).all()

    odd = timestep_embedding(torch.rand(2), 31)
    assert odd.shape == (2, 31)


def test_interpolation_endpoints():
    z0 = torch.randn(2, 64, 8, 8)
    z1 = torch.randn(2, 64, 8, 8)

    assert torch.allclose(RectifiedFlow.interpolate(z0, z1, torch.zeros(2)), z0)
    assert torch.allclose(RectifiedFlow.interpolate(z0, z1, torch.ones(2)), z1)

    mid = RectifiedFlow.interpolate(z0, z1, torch.full((2,), 0.5))
    assert torch.allclose(mid, 0.5 * (z0 + z1), atol=1e-6)


def test_interpolation_is_per_sample_in_t():
    z0 = torch.zeros(2, 4, 2, 2)
    z1 = torch.ones(2, 4, 2, 2)
    out = RectifiedFlow.interpolate(z0, z1, torch.tensor([0.0, 1.0]))

    assert torch.allclose(out[0], torch.zeros(4, 2, 2))
    assert torch.allclose(out[1], torch.ones(4, 2, 2))


def test_velocity_shape():
    model = _rf()
    v = model.velocity(torch.randn(2, 64, 8, 8), torch.rand(2))
    assert v.shape == (2, 64, 8, 8)


def test_zero_init_makes_velocity_zero_and_ode_identity():
    """
    At initialization v_theta == 0, so the ODE is the identity transport. This
    keeps early training stable (see module docstring).
    """
    model = _rf()
    z0 = torch.randn(2, 64, 8, 8)

    assert torch.allclose(model.velocity(z0, torch.rand(2)), torch.zeros_like(z0))
    assert torch.allclose(model.integrate(z0, steps=10), z0, atol=1e-6)


def test_flow_loss_matches_target_velocity_definition():
    """With v_theta == 0 at init, L_flow must equal mean((z1 - z0)^2)."""
    model = _rf()
    z0 = torch.randn(2, 64, 8, 8)
    z1 = torch.randn(2, 64, 8, 8)

    loss, info = model.flow_loss(z0, z1)

    assert torch.allclose(loss, ((z1 - z0) ** 2).mean(), atol=1e-6)
    assert torch.allclose(info["v_target"], z1 - z0)
    assert info["t"].shape == (2,)
    assert ((info["t"] >= 0) & (info["t"] <= 1)).all()


def test_flow_loss_accepts_explicit_t():
    model = _rf()
    z0, z1 = torch.randn(1, 64, 8, 8), torch.randn(1, 64, 8, 8)
    t = torch.tensor([0.3])

    _, info = model.flow_loss(z0, z1, t=t)
    assert torch.allclose(info["t"], t)
    assert torch.allclose(info["z_t"], 0.7 * z0 + 0.3 * z1, atol=1e-6)


def test_flow_loss_gradients():
    model = _rf()
    loss, _ = model.flow_loss(torch.randn(1, 64, 8, 8), torch.randn(1, 64, 8, 8))
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_trajectory_length_and_endpoints():
    model = _rf()
    z0 = torch.randn(1, 64, 8, 8)
    z, traj = model.integrate(z0, steps=5, return_trajectory=True)

    assert len(traj) == 6                      # initial state + 5 steps
    assert torch.allclose(traj[0], z0.float())
    assert torch.allclose(traj[-1], z)


def test_step_count_is_configurable():
    model = _rf(steps=7)
    _, traj = model.integrate(torch.randn(1, 64, 4, 4), return_trajectory=True)
    assert len(traj) == 8

    _, traj_override = model.integrate(torch.randn(1, 64, 4, 4), steps=3, return_trajectory=True)
    assert len(traj_override) == 4


def test_heun_solver_runs():
    model = _rf(solver="heun")
    z = model.integrate(torch.randn(1, 64, 4, 4), steps=4)
    assert z.shape == (1, 64, 4, 4) and torch.isfinite(z).all()


def test_rejects_unknown_solver():
    with pytest.raises(ValueError, match="euler or heun"):
        _rf(solver="rk4")


def test_integration_runs_in_float32_under_half_input():
    """Integration must not accumulate in fp16 (precision loss over many steps)."""
    model = _rf()
    z = model.integrate(torch.randn(1, 64, 4, 4).half(), steps=4)
    assert z.dtype == torch.float32


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
