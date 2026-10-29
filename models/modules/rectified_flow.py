"""
Rectified Flow -- latent-space ODE transport (PROPOSED SRRT component).

Mathematical formulation
------------------------
Rectified Flow (Liu et al., "Flow Straight and Fast: Learning to Generate and
Transfer Data with Rectified Flow", ICLR 2023) learns a velocity field that
transports one distribution to another along straight paths. Applied here in
LATENT space, between the degraded-image latent z0 and the clean-image latent
z1.

Training:
    t   ~ U(0, 1)                       per-sample scalar
    z_t = (1 - t) * z0 + t * z1         linear interpolation
    v*  = z1 - z0                       target velocity (constant along path)
    L_flow = || v_theta(z_t, t) - (z1 - z0) ||_2^2

Inference (z1 unknown): integrate the learned ODE from t=0 to t=1
    dz/dt = v_theta(z, t)
Euler update:
    z_next = z_t + dt * v_theta(z_t, t),   dt = 1/steps

Why rectified flow rather than a generic diffusion sampler
----------------------------------------------------------
For restoration, z0 is an informative starting point (the degraded latent), not
noise. The rectified-flow target velocity is constant along the straight path
from z0 to z1, so a small number of Euler steps (10-20) already tracks the path
closely. A generic DDPM-style sampler would instead start from noise and need
far more steps, discarding information already present in z0. This also keeps
inference cheap enough for a laptop GPU.

Numerical stability
-------------------
  * t is embedded with sinusoidal features (stable across ranges).
  * The velocity network is residual and ends in a ZERO-INITIALIZED conv, so at
    initialization v_theta == 0 and the ODE is the identity map. Training starts
    from a no-op rather than injecting random drift into the backbone; a random
    start would corrupt the latent before the network learns anything.
  * Integration runs in float32 even under AMP, since accumulating many small
    increments in fp16 loses precision.

Shapes:
    z0, z1, z_t, v : (B, latent_dim, h, w)
    t              : (B,) in [0, 1]
"""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.layers import LayerNorm


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of a (B,) float tensor -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class VelocityBlock(nn.Module):
    """Residual conv block with additive time conditioning."""

    def __init__(self, channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = LayerNorm(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.time_proj = nn.Linear(time_dim, channels)
        self.norm2 = LayerNorm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class RectifiedFlow(nn.Module):
    """
    Args:
        latent_dim: channels of the latent the flow operates on.
        hidden_dim: internal width of the velocity network.
        time_embed_dim: width of the sinusoidal time embedding MLP.
        num_blocks: residual blocks in the velocity network.
        steps: default Euler steps used by integrate().
        solver: euler (implemented) or heun (2nd-order predictor-corrector).
    """

    def __init__(
        self,
        latent_dim: int = 384,
        hidden_dim: int = 256,
        time_embed_dim: int = 256,
        num_blocks: int = 3,
        steps: int = 20,
        solver: str = "euler",
    ) -> None:
        super().__init__()

        if solver not in ("euler", "heun"):
            raise ValueError("solver must be euler or heun, got " + repr(solver))

        self.latent_dim = latent_dim
        self.steps = steps
        self.solver = solver
        self.time_embed_dim = time_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.proj_in = nn.Conv2d(latent_dim, hidden_dim, kernel_size=1, bias=False)
        self.blocks = nn.ModuleList(
            [VelocityBlock(hidden_dim, time_embed_dim) for _ in range(num_blocks)]
        )
        self.proj_out = nn.Conv2d(hidden_dim, latent_dim, kernel_size=1, bias=True)

        # Zero-init the final conv so v_theta == 0 at initialization and the ODE
        # starts as the identity transport.
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def velocity(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """v_theta(z, t). z: (B,C,h,w), t: (B,) in [0,1] -> (B,C,h,w)."""
        t_emb = self.time_mlp(timestep_embedding(t, self.time_embed_dim))
        h = self.proj_in(z)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.proj_out(h)

    @staticmethod
    def interpolate(z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """z_t = (1-t) z0 + t z1, broadcasting t over (B,C,h,w)."""
        t_b = t.view(-1, *([1] * (z0.dim() - 1)))
        return (1 - t_b) * z0 + t_b * z1

    def flow_loss(self, z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor | None = None):
        """
        L_flow = || v_theta(z_t,t) - (z1 - z0) ||^2, with t ~ U(0,1) if not given.

        Returns (loss, info) where info holds t, z_t, v_pred and v_target for
        logging and flow visualization.
        """
        if t is None:
            t = torch.rand(z0.shape[0], device=z0.device, dtype=torch.float32)

        z_t = self.interpolate(z0, z1, t)
        v_target = z1 - z0
        v_pred = self.velocity(z_t, t)

        info = {"t": t, "z_t": z_t, "v_pred": v_pred, "v_target": v_target}
        return F.mse_loss(v_pred, v_target), info

    @torch.no_grad()
    def integrate(
        self,
        z0: torch.Tensor,
        steps: int | None = None,
        return_trajectory: bool = False,
    ):
        """Solve dz/dt = v_theta(z,t) from t=0 to t=1 in float32."""
        steps = steps or self.steps
        dt = 1.0 / steps

        z = z0.float()
        trajectory = [z.clone()] if return_trajectory else None

        for i in range(steps):
            t = torch.full((z.shape[0],), i * dt, device=z.device, dtype=torch.float32)
            v = self.velocity(z, t)

            if self.solver == "euler":
                z = z + dt * v
            else:
                z_pred = z + dt * v
                t_next = torch.full_like(t, (i + 1) * dt)
                z = z + dt * 0.5 * (v + self.velocity(z_pred, t_next))

            if return_trajectory:
                trajectory.append(z.clone())

        return (z, trajectory) if return_trajectory else z

    def forward(self, z0: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        return self.integrate(z0, steps)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "RectifiedFlow":
        m = cfg["model"]["modules"]["rectified_flow"]
        return cls(
            latent_dim=cfg["model"].get("latent_dim", 384),
            hidden_dim=m.get("hidden_dim", 256),
            time_embed_dim=m.get("time_embed_dim", 256),
            num_blocks=m.get("num_blocks", 3),
            steps=m.get("steps", 20),
            solver=m.get("solver", "euler"),
        )
