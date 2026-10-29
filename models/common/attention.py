"""
Multi-DConv Head Transposed Attention (MDTA) from Restormer (CVPR 2022).

MEMORY-CRITICAL MODULE -- read before editing.

MDTA applies self-attention across the CHANNEL dimension, not the spatial one.
For X in R^(B x C x H x W) with h heads and d = C/h channels per head:

    Q, K, V = DWConv3x3(Conv1x1(X)),  each (B, C, H, W)
    reshape each to (B, h, d, HW)
    Q, K are L2-normalized along the HW axis
    A    = Softmax( (Q @ K^T) * alpha )     -> (B, h, d, d)
    Out  = Conv1x1( (A @ V) reshaped back ) -> (B, C, H, W)

with alpha a learnable per-head temperature.

Complexity:
    naive spatial attention : O((HW)^2 * C)   -> matrix is HW x HW
    MDTA channel attention  : O(HW * C^2)     -> matrix is d x d

At H=W=256, C=48: HW = 65,536 so an HW x HW float32 matrix would be
65,536^2 = 4,294,967,296 elements ~= 17 GB per head (and ~34 GB was
observed in practice with intermediates). The d x d matrix is 48x48.

A previous revision of this file computed `q.transpose(-2,-1) @ k`, which
collapses the CHANNEL axis and leaves an (HW x HW) matrix -- that is spatial
attention and it exhausted system memory. The correct order below,
`q @ k.transpose(-2,-1)`, collapses the SPATIAL axis instead. Do not swap it.
tests/test_attention.py guards this with an explicit regression test.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MDTA(nn.Module):
    """
    Args:
        dim: number of channels C. Must be divisible by num_heads.
        num_heads: number of attention heads h.
        bias: whether the 1x1 / depthwise convolutions carry a bias term.

    Shapes:
        input : (B, dim, H, W)
        output: (B, dim, H, W)
    """

    def __init__(self, dim: int, num_heads: int, bias: bool = False) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads

        # Learnable per-head temperature (alpha). Initialized to 1.
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 1x1 pointwise projection to Q, K, V simultaneously.
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        # 3x3 depthwise convolution, giving each of Q, K, V local spatial context.
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        head_dim = c // self.num_heads

        # (B, C, H, W) -> (B, heads, head_dim, HW)
        q = q.reshape(b, self.num_heads, head_dim, h * w)
        k = k.reshape(b, self.num_heads, head_dim, h * w)
        v = v.reshape(b, self.num_heads, head_dim, h * w)

        # Normalize along the spatial axis so the dot product below behaves
        # like a cosine similarity between channel response maps.
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # (B,h,d,HW) @ (B,h,HW,d) -> (B,h,d,d). Channel attention: the HW axis
        # is contracted away. See module docstring -- never reverse this.
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # (B,h,d,d) @ (B,h,d,HW) -> (B,h,d,HW)
        out = torch.matmul(attn, v)
        out = out.reshape(b, c, h, w)

        return self.project_out(out)
