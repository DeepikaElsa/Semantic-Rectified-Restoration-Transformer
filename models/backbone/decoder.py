"""Hierarchical Restormer decoder (3 upsampling levels with skip connections)."""

from typing import Sequence

import torch
import torch.nn as nn

from models.common.blocks import TransformerStage


class Upsample(nn.Module):
    """
    Restormer upsample: double H,W and halve C via Conv + PixelShuffle.

    Conv maps C -> 2C, then PixelShuffle(2) turns (2C, H, W) into
    (C/2, 2H, 2W).

    Shapes:
        input : (B, C,   H,  W)
        output: (B, C/2, 2H, 2W)
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Decoder(nn.Module):
    """
    Three-level hierarchical decoder with encoder skip connections.

    Design note (faithful to the original Restormer): levels 3 and 2 reduce the
    concatenated channels back down with a 1x1 conv, but level 1 deliberately
    does NOT -- it keeps the full dim*2 width, preserving extra capacity at the
    highest resolution. The refinement stage downstream therefore also operates
    at dim*2.

    Shapes for dim=48, input (B, 3, H, W) to the network:
        latent (B, 384, H/8, W/8) + skips -> output (B, 96, H, W)
    """

    def __init__(
        self,
        dim: int = 48,
        decoder_blocks: Sequence[int] = (6, 6, 4),
        heads: Sequence[int] = (1, 2, 4, 8),
        expansion_factor: float = 2.66,
        layer_norm_type: str = "WithBias",
    ) -> None:
        super().__init__()

        if len(decoder_blocks) != 3:
            raise ValueError(f"decoder_blocks must have 3 entries, got {len(decoder_blocks)}")
        if len(heads) != 4:
            raise ValueError(f"heads must have 4 entries (shared with encoder), got {len(heads)}")

        # Level 3: latent (dim*8) -> dim*4, concat skip x3 (dim*4) -> reduce -> dim*4
        self.up3 = Upsample(dim * 8)
        self.reduce3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=False)
        self.level3 = TransformerStage(dim * 4, decoder_blocks[0], heads[2], expansion_factor, layer_norm_type)

        # Level 2: dim*4 -> dim*2, concat skip x2 (dim*2) -> reduce -> dim*2
        self.up2 = Upsample(dim * 4)
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=False)
        self.level2 = TransformerStage(dim * 2, decoder_blocks[1], heads[1], expansion_factor, layer_norm_type)

        # Level 1: dim*2 -> dim, concat skip x1 (dim) -> dim*2 (no reduction)
        self.up1 = Upsample(dim * 2)
        self.level1 = TransformerStage(dim * 2, decoder_blocks[2], heads[0], expansion_factor, layer_norm_type)

    def forward(
        self,
        latent: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
    ) -> torch.Tensor:
        d3 = self.level3(self.reduce3(torch.cat([self.up3(latent), x3], dim=1)))
        d2 = self.level2(self.reduce2(torch.cat([self.up2(d3), x2], dim=1)))
        d1 = self.level1(torch.cat([self.up1(d2), x1], dim=1))
        return d1
