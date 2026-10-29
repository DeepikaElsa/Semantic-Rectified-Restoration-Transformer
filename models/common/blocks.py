"""Transformer building blocks shared by the backbone and SRRT modules."""

import torch
import torch.nn as nn

from models.common.attention import MDTA
from models.common.feedforward import GDFN
from models.common.layers import LayerNorm


class TransformerBlock(nn.Module):
    """
    Restormer transformer block.

        X' = X + MDTA(LN(X))
        Y  = X' + GDFN(LN(X'))

    Shapes:
        input : (B, dim, H, W)
        output: (B, dim, H, W)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        expansion_factor: float = 2.66,
        layer_norm_type: str = "WithBias",
    ) -> None:
        super().__init__()

        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.attn = MDTA(dim, num_heads)

        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.ffn = GDFN(dim, expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerStage(nn.Module):
    """
    A stack of `num_blocks` TransformerBlocks at a single resolution.

    Used by the encoder levels, the bottleneck, the decoder levels and the
    refinement stage, so the block-stacking logic lives in exactly one place.

    Shapes:
        input : (B, dim, H, W)
        output: (B, dim, H, W)
    """

    def __init__(
        self,
        dim: int,
        num_blocks: int,
        num_heads: int,
        expansion_factor: float = 2.66,
        layer_norm_type: str = "WithBias",
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    expansion_factor=expansion_factor,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
