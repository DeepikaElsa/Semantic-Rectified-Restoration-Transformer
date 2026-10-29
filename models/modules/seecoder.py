"""
SeeCoder -- visual-context encoder (PROPOSED SRRT component).

Mathematical purpose
--------------------
    F_v = E_v(y; theta_v)

Extracts multi-scale *visual/spatial context* from the degraded input y.
"Context" here means mid-level structural information (edges, texture layout,
object extent) that the restoration backbone can condition on, as opposed to
the degradation-specific cues that DiffAE targets.

Design rationale (documented assumption)
----------------------------------------
This is deliberately NOT a second copy of Restormer. Duplicating the backbone
would (a) double parameters/VRAM for no clear reason and (b) make any ablation
gain uninterpretable -- you could not tell whether a gain came from "visual
context" or simply from more Restormer capacity.

Instead SeeCoder is a lightweight convolutional pyramid:
  * depthwise-separable conv blocks (cheap, strong local inductive bias),
  * strided downsampling to 1/2, 1/4, 1/8 resolution,
  * ONE channel-attention (MDTA) block at the deepest level only, to mix global
    channel statistics without any HW x HW matrix.

Shapes (dim=48, embed_dim=256, input (B,3,H,W)); H,W divisible by 8:
    features[0] : (B,  48, H/2, W/2)
    features[1] : (B,  96, H/4, W/4)
    features[2] : (B, 192, H/8, W/8)
    context     : (B, 256, H/8, W/8)     <- embed_dim, consumed by LCR
    global_vec  : (B, 256)               <- spatially pooled context

Learnable parameters: conv stem, 3 pyramid stages, 1 MDTA block, 1x1 context
projection. No pretrained weights are used.
"""

from typing import Any

import torch
import torch.nn as nn

from models.common.attention import MDTA
from models.common.layers import LayerNorm


class SeparableConvBlock(nn.Module):
    """Depthwise 3x3 -> pointwise 1x1 -> LN -> GELU, with residual when shapes allow."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        self.dwconv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False,
        )
        self.pwconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = LayerNorm(out_channels)
        self.act = nn.GELU()
        self.use_residual = (stride == 1 and in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm(self.pwconv(self.dwconv(x))))
        return x + out if self.use_residual else out


class SeeCoder(nn.Module):
    """
    Args:
        in_channels: input image channels.
        dim: base width; stage widths are dim, dim*2, dim*4.
        embed_dim: width of the emitted context tensor (what LCR consumes).
        blocks_per_stage: residual blocks after each downsample.
        num_heads: heads for the single deepest-level MDTA block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 48,
        embed_dim: int = 256,
        blocks_per_stage: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.embed_dim = embed_dim
        self.out_stride = 8

        self.stem = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1, bias=False)

        widths = (dim, dim * 2, dim * 4)
        self.stages = nn.ModuleList()
        prev = dim

        for width in widths:
            stage = [SeparableConvBlock(prev, width, stride=2)]
            stage += [SeparableConvBlock(width, width) for _ in range(blocks_per_stage)]
            self.stages.append(nn.Sequential(*stage))
            prev = width

        # Channel attention at the deepest (cheapest) level only.
        deepest = widths[-1]
        if deepest % num_heads != 0:
            num_heads = 1
        self.norm = LayerNorm(deepest)
        self.attn = MDTA(deepest, num_heads)

        self.to_context = nn.Conv2d(deepest, embed_dim, kernel_size=1, bias=False)

    def forward(self, y: torch.Tensor) -> dict[str, Any]:
        x = self.stem(y)

        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        x = x + self.attn(self.norm(x))
        context = self.to_context(x)

        return {
            "features": features,
            "context": context,
            "global_vec": context.mean(dim=(2, 3)),
        }

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SeeCoder":
        m = cfg["model"]["modules"]["seecoder"] if "model" in cfg else cfg
        r = cfg["model"]["restormer"] if "model" in cfg else {}
        return cls(
            in_channels=r.get("in_channels", 3),
            dim=m.get("dim", r.get("dim", 48)),
            embed_dim=m.get("embed_dim", 256),
            blocks_per_stage=m.get("blocks_per_stage", 2),
            num_heads=m.get("num_heads", 4),
        )
