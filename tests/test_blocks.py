import torch

from models.common.blocks import TransformerBlock
from models.common.attention import MDTA
from models.common.feedforward import GDFN
from models.common.layers import LayerNorm


def test_transformer_block_shape_small():
    x = torch.randn(2, 48, 16, 16)

    block = TransformerBlock(dim=48, num_heads=4)
    y = block(x)

    assert y.shape == x.shape


def test_transformer_block_uses_tested_submodules():
    block = TransformerBlock(dim=48, num_heads=4)

    assert isinstance(block.norm1, LayerNorm)
    assert isinstance(block.norm2, LayerNorm)
    assert isinstance(block.attn, MDTA)
    assert isinstance(block.ffn, GDFN)


def test_transformer_block_realistic_resolution():
    x = torch.randn(1, 48, 256, 256)

    block = TransformerBlock(dim=48, num_heads=1)
    y = block(x)

    assert y.shape == x.shape


if __name__ == "__main__":
    test_transformer_block_shape_small()
    test_transformer_block_uses_tested_submodules()
    test_transformer_block_realistic_resolution()
    print("All block tests passed.")
