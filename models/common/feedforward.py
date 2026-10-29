"""Gated-Dconv Feed-Forward Network (GDFN) from Restormer (CVPR 2022)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GDFN(nn.Module):
    """
    Gated depthwise-convolutional feed-forward network.

        X1       = Conv1x1(X)                  -> 2 * hidden channels
        X2       = DWConv3x3(X1)
        Xa, Xb   = split(X2)                   -> hidden each
        G        = GELU(Xa) * Xb               (elementwise gate)
        Y        = Conv1x1(G)                  -> dim

    where hidden = int(dim * expansion_factor).

    The gating lets the network suppress less informative features before the
    output projection, and the depthwise conv supplies local spatial context.

    Args:
        dim: input/output channels.
        expansion_factor: hidden width multiplier (Restormer default 2.66).
        bias: whether the convolutions carry a bias term.

    Shapes:
        input : (B, dim, H, W)
        output: (B, dim, H, W)
    """

    def __init__(self, dim: int, expansion_factor: float = 2.66, bias: bool = False) -> None:
        super().__init__()

        hidden_features = int(dim * expansion_factor)
        self.hidden_features = hidden_features

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwconv(self.project_in(x))
        x1, x2 = x.chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)
