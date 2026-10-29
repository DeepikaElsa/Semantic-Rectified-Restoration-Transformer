"""
LCR -- Latent Consistency Refinement (PRIMARY PROPOSED NOVELTY of SRRT).

Mathematical purpose
--------------------
Fuse the visual-context latent from SeeCoder with the damage-aware latent from
DiffAE into a single initial latent:

    z_0 = L(F_v, F_d; theta_L)

and expose an explicit latent *consistency* objective

    L_LCR = || P(z_pred) - P(z_ref) ||_2^2

where P is a shared, learnable projection head applied to both arguments and
z_ref is a reference latent (in SRRT: the latent obtained by encoding the clean
target image with the same backbone encoder; available only during training).
Both projections are L2-normalized before the distance so the loss measures
*directional* agreement and cannot be trivially minimized by shrinking latent
magnitudes to zero -- a standard collapse failure mode for consistency losses.

Fusion mechanism
----------------
1. Project F_v and F_d to a common width `hidden_dim` (they arrive with
   different channel counts and, in general, different spatial sizes; F_d is
   bilinearly resized to F_v's grid when needed).
2. Concatenate and mix with a 1x1 conv (gives the network an explicit,
   learnable weighting between visual and damage evidence).
3. Cross-conditioning via CHANNEL attention (MDTA), never spatial attention --
   an HW x HW map here would reintroduce the original memory blow-up.
4. `refinement_blocks` residual transformer blocks.
5. 1x1 projection to `latent_dim` (the backbone bottleneck width) so z_0 can be
   injected into Restormer without a shape mismatch.

STATUS: implemented and unit-tested. NOT trained, NOT experimentally validated.
No claim is made here that LCR improves restoration quality; that is exactly
what the ablation study (configs with modules.lcr.enabled true/false) exists to
measure.

Shapes (hidden_dim=384, latent_dim=384):
    F_v      : (B, C_v, h, w)
    F_d      : (B, C_d, h', w')   resized to (h, w) if it differs
    z_0      : (B, latent_dim, h, w)
    P(z)     : (B, proj_dim)      pooled + projected, L2-normalized
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.blocks import TransformerBlock
from models.common.attention import MDTA
from models.common.layers import LayerNorm


class LCR(nn.Module):
    """
    Args:
        visual_dim: channels of the SeeCoder context tensor F_v.
        damage_dim: channels of the DiffAE spatial latent F_d.
        hidden_dim: internal fusion width.
        latent_dim: output width (must match the backbone bottleneck).
        proj_dim: width of the consistency projection head P.
        num_heads: heads for the channel-attention fusion / refinement blocks.
        refinement_blocks: number of residual transformer blocks after fusion.
        dropout: applied inside the projection head only.
    """

    def __init__(
        self,
        visual_dim: int = 256,
        damage_dim: int = 256,
        hidden_dim: int = 384,
        latent_dim: int = 384,
        proj_dim: int = 384,
        num_heads: int = 8,
        refinement_blocks: int = 2,
        dropout: float = 0.0,
        expansion_factor: float = 2.66,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.proj_v = nn.Conv2d(visual_dim, hidden_dim, kernel_size=1, bias=False)
        self.proj_d = nn.Conv2d(damage_dim, hidden_dim, kernel_size=1, bias=False)

        # Explicit learnable weighting between visual and damage evidence.
        self.fuse = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=False)

        self.norm_fuse = LayerNorm(hidden_dim)
        self.cross_attn = MDTA(hidden_dim, num_heads)

        self.refine = nn.Sequential(*[
            TransformerBlock(hidden_dim, num_heads, expansion_factor)
            for _ in range(refinement_blocks)
        ])

        self.to_latent = nn.Conv2d(hidden_dim, latent_dim, kernel_size=1, bias=False)

        # Shared projection head P used on BOTH sides of the consistency loss.
        self.project = nn.Sequential(
            nn.Linear(latent_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, visual: torch.Tensor, damage: torch.Tensor) -> dict[str, torch.Tensor]:
        v = self.proj_v(visual)
        d = self.proj_d(damage)

        if d.shape[-2:] != v.shape[-2:]:
            d = F.interpolate(d, size=v.shape[-2:], mode="bilinear", align_corners=False)

        z = self.fuse(torch.cat([v, d], dim=1))
        z = z + self.cross_attn(self.norm_fuse(z))
        z = self.refine(z)

        z0 = self.to_latent(z)
        return {"z0": z0, "fused": z}

    def project_latent(self, z: torch.Tensor) -> torch.Tensor:
        """P(z): global-average-pool -> MLP -> L2 normalize. Shape (B, proj_dim)."""
        return F.normalize(self.project(z.mean(dim=(2, 3))), dim=-1)

    def consistency_loss(self, z_pred: torch.Tensor, z_ref: torch.Tensor) -> torch.Tensor:
        """
        L_LCR = || P(z_pred) - P(z_ref) ||^2, averaged over the batch.

        z_ref is treated as a fixed target (detached): the reference latent
        comes from the clean image and should pull z_pred toward it, not be
        dragged toward z_pred.
        """
        return F.mse_loss(self.project_latent(z_pred), self.project_latent(z_ref.detach()))

    @classmethod
    def from_config(cls, cfg: dict[str, Any], visual_dim: int, damage_dim: int) -> "LCR":
        m = cfg["model"]["modules"]["lcr"]
        return cls(
            visual_dim=visual_dim,
            damage_dim=damage_dim,
            hidden_dim=m.get("hidden_dim", 384),
            latent_dim=cfg["model"].get("latent_dim", 384),
            proj_dim=m.get("proj_dim", 384),
            num_heads=m.get("num_heads", 8),
            refinement_blocks=m.get("refinement_blocks", 2),
            dropout=m.get("dropout", 0.0),
            expansion_factor=cfg["model"]["restormer"].get("expansion_factor", 2.66),
        )
