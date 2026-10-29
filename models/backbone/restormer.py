"""
Restormer backbone (Zamir et al., "Restormer: Efficient Transformer for
High-Resolution Image Restoration", CVPR 2022).

This is the project's BASELINE. It is implemented to follow the published
formulation; any deliberate deviation is documented in docs/design_decisions.md.

Pipeline:
    y -> OverlapPatchEmbed -> Encoder(3 levels) -> Bottleneck
      -> Decoder(3 levels, skip connections) -> Refinement -> 3x3 conv
      -> + y (global residual) -> x_hat
"""

from typing import Any, Sequence

import torch
import torch.nn as nn

from models.backbone.bottleneck import Bottleneck
from models.backbone.decoder import Decoder
from models.backbone.encoder import Encoder
from models.common.blocks import TransformerStage


class Restormer(nn.Module):
    """
    Args mirror configs/srrt_base.yaml -> model.restormer. Use `from_config`
    to build straight from a loaded YAML dict.

    Note on `encoder_blocks`: the YAML lists 4 entries, e.g. [4, 6, 6, 8].
    The first three are the encoder levels; the fourth is the bottleneck
    depth. Likewise `num_heads` has 4 entries, the last being the bottleneck.

    Shapes (dim=48):
        input : (B, 3, H, W)   H, W divisible by 8
        output: (B, 3, H, W)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        encoder_blocks: Sequence[int] = (4, 6, 6, 8),
        decoder_blocks: Sequence[int] = (6, 6, 4),
        heads: Sequence[int] = (1, 2, 4, 8),
        expansion_factor: float = 2.66,
        refinement_blocks: int = 4,
        layer_norm_type: str = "WithBias",
    ) -> None:
        super().__init__()

        if len(encoder_blocks) != 4:
            raise ValueError(
                f"encoder_blocks must have 4 entries (3 levels + bottleneck), got {len(encoder_blocks)}"
            )

        self.dim = dim
        self.latent_dim = dim * 8

        self.encoder = Encoder(
            in_channels=in_channels,
            dim=dim,
            encoder_blocks=encoder_blocks[:3],
            heads=heads,
            expansion_factor=expansion_factor,
            layer_norm_type=layer_norm_type,
        )

        self.bottleneck = Bottleneck(
            dim=dim,
            num_blocks=encoder_blocks[3],
            num_heads=heads[3],
            expansion_factor=expansion_factor,
            layer_norm_type=layer_norm_type,
        )

        self.decoder = Decoder(
            dim=dim,
            decoder_blocks=decoder_blocks,
            heads=heads,
            expansion_factor=expansion_factor,
            layer_norm_type=layer_norm_type,
        )

        # Refinement runs at dim*2 because decoder level 1 keeps the
        # concatenated width (see Decoder docstring).
        self.refinement = TransformerStage(
            dim=dim * 2,
            num_blocks=refinement_blocks,
            num_heads=heads[0],
            expansion_factor=expansion_factor,
            layer_norm_type=layer_norm_type,
        )

        self.output = nn.Conv2d(dim * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Full forward exposing every intermediate tensor.

        Returned keys (dim=48, input (B,3,H,W)):
            skip1     (B,  48, H,   W)
            skip2     (B,  96, H/2, W/2)
            skip3     (B, 192, H/4, W/4)
            latent    (B, 384, H/8, W/8)   bottleneck output
            decoded   (B,  96, H,   W)
            refined   (B,  96, H,   W)
            residual  (B,   3, H,   W)     predicted residual before += input
            output    (B,   3, H,   W)
        """
        skip1, skip2, skip3, x4 = self.encoder(x)
        latent = self.bottleneck(x4)

        decoded = self.decoder(latent, skip1, skip2, skip3)
        refined = self.refinement(decoded)

        residual = self.output(refined)

        return {
            "skip1": skip1,
            "skip2": skip2,
            "skip3": skip3,
            "latent": latent,
            "decoded": decoded,
            "refined": refined,
            "residual": residual,
            "output": residual + x,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)["output"]

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Restormer":
        """
        Build from a loaded YAML config. Accepts either the full config dict
        or just the `model.restormer` sub-dict.
        """
        if "model" in cfg:
            r = cfg["model"]["restormer"]
        elif "restormer" in cfg:
            r = cfg["restormer"]
        else:
            r = cfg

        return cls(
            in_channels=r.get("in_channels", 3),
            out_channels=r.get("out_channels", 3),
            dim=r.get("dim", 48),
            encoder_blocks=tuple(r.get("encoder_blocks", (4, 6, 6, 8))),
            decoder_blocks=tuple(r.get("decoder_blocks", (6, 6, 4))),
            heads=tuple(r.get("num_heads", (1, 2, 4, 8))),
            expansion_factor=r.get("expansion_factor", 2.66),
            refinement_blocks=r.get("refinement_blocks", 4),
            layer_norm_type=r.get("layer_norm_type", "WithBias"),
        )
