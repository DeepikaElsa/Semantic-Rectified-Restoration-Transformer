"""
Normalization layers and tensor-layout helpers.

The two LayerNorm variants follow Restormer: 'BiasFree' normalizes by variance
only (no mean subtraction, no bias), 'WithBias' is standard LayerNorm. Both
normalize over the CHANNEL dimension of a 2D feature map.

The math here is unchanged from the original tested implementation; only type
hints and documentation were added.
"""

import torch
import torch.nn as nn


def to_3d(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H*W, C)"""
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)


def to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(B, H*W, C) -> (B, C, H, W)"""
    b, _, c = x.shape
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2)


class BiasFreeLayerNorm(nn.Module):
    """
    Variance-only normalization: x / sqrt(var(x) + eps) * weight.

    No mean subtraction and no learnable bias. Restormer uses this variant for
    tasks where preserving the input's mean intensity matters.

    Shapes:
        input : (..., normalized_shape)
        output: same
    """

    def __init__(self, normalized_shape: int | tuple[int, ...], eps: float = 1e-5) -> None:
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + self.eps) * self.weight


class WithBiasLayerNorm(nn.Module):
    """
    Standard LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias.

    Shapes:
        input : (..., normalized_shape)
        output: same
    """

    def __init__(self, normalized_shape: int | tuple[int, ...], eps: float = 1e-5) -> None:
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + self.eps) * self.weight + self.bias


class LayerNorm(nn.Module):
    """
    Channel-wise LayerNorm for 2D feature maps.

    Internally flattens (B, C, H, W) -> (B, HW, C), normalizes over C, and
    restores the original layout.

    Args:
        dim: number of channels C.
        layer_type: 'WithBias' (default) or 'BiasFree'.

    Shapes:
        input : (B, dim, H, W)
        output: (B, dim, H, W)
    """

    def __init__(self, dim: int, layer_type: str = "WithBias") -> None:
        super().__init__()

        if layer_type not in ("BiasFree", "WithBias"):
            raise ValueError(f"layer_type must be 'BiasFree' or 'WithBias', got {layer_type!r}")

        self.norm = BiasFreeLayerNorm(dim) if layer_type == "BiasFree" else WithBiasLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return to_4d(self.norm(to_3d(x)), h, w)
