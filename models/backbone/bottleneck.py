"""
Restormer bottleneck (latent) stage.

Factored into its own module because it is the natural injection point for the
SRRT latent pipeline (LCR / Rectified Flow operate on the deepest, lowest
resolution representation). Keeping it separate means those modules can be
attached or ablated without editing the encoder or decoder.
"""

import torch
import torch.nn as nn

from models.common.blocks import TransformerStage


class Bottleneck(nn.Module):
    """
    Transformer stack at the deepest level (dim*8, H/8, W/8).

    Shapes for dim=48, input (B, 3, H, W) to the network:
        input : (B, 384, H/8, W/8)
        output: (B, 384, H/8, W/8)
    """

    def __init__(
        self,
        dim: int = 48,
        num_blocks: int = 8,
        num_heads: int = 8,
        expansion_factor: float = 2.66,
        layer_norm_type: str = "WithBias",
    ) -> None:
        super().__init__()

        self.latent_dim = dim * 8

        self.stage = TransformerStage(
            dim=self.latent_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            expansion_factor=expansion_factor,
            layer_norm_type=layer_norm_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage(x)
