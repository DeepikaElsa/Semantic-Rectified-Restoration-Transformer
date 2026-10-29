"""
SRA -- Self-Rectifying Attention (PROPOSED SRRT component).

Mathematical formulation
------------------------
Given a restored/refined feature map F_R produced by the backbone decoder, SRA
predicts a residual correction and applies it under a learned gate:

    A_s     = sigma( W_a * MDTA( LN(F_R) ) )      attention/gating map, in (0,1)
    R(F_R)  = W_r2 * phi( DWConv( W_r1 * LN(F_R) ) )   residual correction
    F_rect  = F_R + gamma * ( A_s (*) R(F_R) )    (*) = elementwise product

where sigma is the sigmoid, phi is GELU, and gamma is a learnable scalar
initialized to ZERO.

Design decisions (documented, not established literature)
---------------------------------------------------------
1. "Self-rectifying" = the module derives BOTH the correction and the gate that
   admits it from the same feature map F_R, with no external conditioning. The
   network decides where its own output still needs fixing.

2. The gate A_s is a per-channel-per-position sigmoid map, so the correction can
   be applied selectively (e.g. only near motion-blurred edges) instead of
   uniformly. A uniform correction would be equivalent to just adding more
   convolution layers, which is not the intended mechanism.

3. Attention is computed with MDTA (channel/transposed attention). Spatial
   attention here would build an HW x HW matrix at FULL decoder resolution --
   the exact failure that crashed this project earlier. Never substitute a
   spatial attention here.

4. gamma is zero-initialized, so at the start of training F_rect == F_R exactly
   and SRA is a strict no-op. This makes the ablation honest: enabling SRA
   cannot degrade a converged backbone merely by perturbing it at init, and any
   measured change is attributable to what SRA actually learned.

STATUS: implemented and unit-tested. NOT trained, NOT experimentally validated.

Shapes:
    F_R    : (B, dim, H, W)
    A_s    : (B, dim, H, W)   in (0,1)
    R(F_R) : (B, dim, H, W)
    F_rect : (B, dim, H, W)
"""

from typing import Any

import torch
import torch.nn as nn

from models.common.attention import MDTA
from models.common.layers import LayerNorm


class SRA(nn.Module):
    """
    Args:
        dim: channels of the incoming feature map F_R.
        num_heads: heads for the internal channel attention.
        expansion: width multiplier of the residual-correction branch.
    """

    def __init__(self, dim: int, num_heads: int = 4, expansion: float = 1.0) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        hidden = max(int(dim * expansion), 1)

        # --- attention / gating branch -> A_s ---
        self.norm_attn = LayerNorm(dim)
        self.attn = MDTA(dim, num_heads)
        self.to_gate = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        # --- residual correction branch -> R(F_R) ---
        self.norm_res = LayerNorm(dim)
        self.res_in = nn.Conv2d(dim, hidden, kernel_size=1, bias=False)
        self.res_dw = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False)
        self.res_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)
        self.act = nn.GELU()

        # Zero-initialized gain: SRA is an exact no-op at initialization.
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat: torch.Tensor, return_maps: bool = False):
        attention = torch.sigmoid(self.to_gate(self.attn(self.norm_attn(feat))))

        residual = self.res_out(self.act(self.res_dw(self.res_in(self.norm_res(feat)))))

        rectified = feat + self.gamma * (attention * residual)

        if return_maps:
            return rectified, {"attention": attention, "residual": residual}
        return rectified

    @classmethod
    def from_config(cls, cfg: dict[str, Any], dim: int) -> "SRA":
        m = cfg["model"]["modules"]["sra"]
        num_heads = m.get("num_heads", 4)
        if dim % num_heads != 0:
            num_heads = 1
        return cls(dim=dim, num_heads=num_heads, expansion=m.get("expansion", 1.0))
