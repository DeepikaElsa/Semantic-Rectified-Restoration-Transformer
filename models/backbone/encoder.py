"""Hierarchical Restormer encoder (overlap patch embed + 3 downsampling levels)."""

from typing import Sequence

import torch
import torch.nn as nn

from models.common.blocks import TransformerStage


class OverlapPatchEmbed(nn.Module):
    """
    Overlapping patch embedding: 3x3 stride-1 conv, RGB -> `dim` channels.

    Shapes:
        input : (B, in_channels, H, W)
        output: (B, dim, H, W)
    """

    def __init__(self, in_channels: int = 3, dim: int = 48, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    """
    Restormer downsample: halve H,W and double C via Conv + PixelUnshuffle.

    Conv maps C -> C//2, then PixelUnshuffle(2) turns (C//2, H, W) into
    (2C, H/2, W/2).

    Shapes:
        input : (B, C,  H,   W)
        output: (B, 2C, H/2, W/2)
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Encoder(nn.Module):
    """
    Three-level hierarchical encoder.

    All dimensions derive from `dim`; nothing is hardcoded (values come from
    configs/srrt_base.yaml via Restormer.from_config).

    Shapes for dim=48, input (B, 3, H, W):
        x1 : (B,  48, H,   W)     level 1 skip
        x2 : (B,  96, H/2, W/2)   level 2 skip
        x3 : (B, 192, H/4, W/4)   level 3 skip
        x4 : (B, 384, H/8, W/8)   -> fed to Bottleneck

    Note: H and W must be divisible by 8 (three stride-2 stages).
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 48,
        encoder_blocks: Sequence[int] = (4, 6, 6),
        heads: Sequence[int] = (1, 2, 4, 8),
        expansion_factor: float = 2.66,
        layer_norm_type: str = "WithBias",
    ) -> None:
        super().__init__()

        if len(encoder_blocks) != 3:
            raise ValueError(f"encoder_blocks must have 3 entries (levels 1-3), got {len(encoder_blocks)}")
        if len(heads) != 4:
            raise ValueError(f"heads must have 4 entries (3 levels + bottleneck), got {len(heads)}")

        self.patch_embed = OverlapPatchEmbed(in_channels, dim)

        self.level1 = TransformerStage(dim, encoder_blocks[0], heads[0], expansion_factor, layer_norm_type)
        self.down1 = Downsample(dim)

        self.level2 = TransformerStage(dim * 2, encoder_blocks[1], heads[1], expansion_factor, layer_norm_type)
        self.down2 = Downsample(dim * 2)

        self.level3 = TransformerStage(dim * 4, encoder_blocks[2], heads[2], expansion_factor, layer_norm_type)
        self.down3 = Downsample(dim * 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.level1(self.patch_embed(x))
        x2 = self.level2(self.down1(x1))
        x3 = self.level3(self.down2(x2))
        x4 = self.down3(x3)
        return x1, x2, x3, x4
