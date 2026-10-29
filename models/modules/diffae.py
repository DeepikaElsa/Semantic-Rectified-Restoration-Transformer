"""
DiffAE -- damage-aware semantic encoder (PROPOSED SRRT component).

Mathematical purpose
--------------------
    F_d = E_d(y; theta_d)

Encodes degradation-related information from the degraded input y into a latent
representation.

IMPORTANT INTERPRETATION CAVEAT
------------------------------
This module produces a *damage-aware learned representation*. It does NOT
provably isolate degradation from content: nothing in the architecture or the
training objective enforces a disentanglement guarantee, and no experiment in
this repository has demonstrated one. Any claim stronger than "a learned
representation that the network is free to use for degradation cues" would be
unsupported. See docs/design_decisions.md.

The name follows the diffusion-autoencoder family (an encoder yielding a
compact semantic code plus a spatial code). The optional decoder head makes the
"AE" concrete: it can reconstruct y from the latent, giving a self-supervised
signal that encourages the latent to actually retain input information rather
than collapsing. The reconstruction term is OFF by default and must be enabled
explicitly, since it has not been validated.

Shapes (latent_dim=256, input (B,3,H,W)); H,W divisible by 8:
    spatial   : (B, 256, H/8, W/8)   spatial damage-aware feature map
    semantic  : (B, 256)             globally pooled semantic code
    recon     : (B, 3, H, W)         only when return_reconstruction=True

Learnable parameters: 3 strided encoder stages, 1x1 latent projection,
semantic MLP head, and (optional) a small decoder.
"""

from typing import Any

import torch
import torch.nn as nn

from models.common.layers import LayerNorm


class EncBlock(nn.Module):
    """Strided 3x3 conv -> LN -> GELU -> 3x3 conv -> LN -> GELU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            LayerNorm(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            LayerNorm(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class DiffAE(nn.Module):
    """
    Args:
        in_channels: input image channels.
        dim: base width; stages are dim, dim*2, dim*4.
        latent_dim: width of the emitted spatial/semantic latent.
        with_decoder: build the optional reconstruction decoder.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 32,
        latent_dim: int = 256,
        with_decoder: bool = False,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.out_stride = 8
        self.with_decoder = with_decoder

        self.stem = nn.Conv2d(in_channels, dim, 3, stride=1, padding=1, bias=False)
        self.enc1 = EncBlock(dim, dim * 2)
        self.enc2 = EncBlock(dim * 2, dim * 4)
        self.enc3 = EncBlock(dim * 4, dim * 8)

        self.to_latent = nn.Conv2d(dim * 8, latent_dim, kernel_size=1, bias=False)

        self.semantic_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        if with_decoder:
            self.decoder = nn.Sequential(
                nn.Conv2d(latent_dim, dim * 8, 3, padding=1, bias=False), LayerNorm(dim * 8), nn.GELU(),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(dim * 8, dim * 4, 3, padding=1, bias=False), LayerNorm(dim * 4), nn.GELU(),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(dim * 4, dim * 2, 3, padding=1, bias=False), LayerNorm(dim * 2), nn.GELU(),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(dim * 2, in_channels, 3, padding=1, bias=False),
            )

    def forward(self, y: torch.Tensor, return_reconstruction: bool = False) -> dict[str, Any]:
        x = self.enc3(self.enc2(self.enc1(self.stem(y))))
        spatial = self.to_latent(x)

        out: dict[str, Any] = {
            "spatial": spatial,
            "semantic": self.semantic_head(spatial.mean(dim=(2, 3))),
        }

        if return_reconstruction:
            if not self.with_decoder:
                raise RuntimeError(
                    "DiffAE was built with with_decoder=False; cannot return a reconstruction."
                )
            out["recon"] = self.decoder(spatial)

        return out

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "DiffAE":
        m = cfg["model"]["modules"]["diffae"] if "model" in cfg else cfg
        r = cfg["model"]["restormer"] if "model" in cfg else {}
        return cls(
            in_channels=r.get("in_channels", 3),
            dim=m.get("dim", 32),
            latent_dim=m.get("latent_dim", 256),
            with_decoder=m.get("with_decoder", False),
        )
