import torch

from models.common.layers import to_3d, to_4d, BiasFreeLayerNorm, WithBiasLayerNorm, LayerNorm


def test_to_3d_to_4d_roundtrip():
    x = torch.randn(2, 16, 8, 10)

    y = to_3d(x)
    assert y.shape == (2, 80, 16)

    z = to_4d(y, 8, 10)
    assert z.shape == x.shape
    assert torch.allclose(z, x)


def test_bias_free_layer_norm_shape_and_stats():
    x = torch.randn(2, 100, 16) * 5 + 3

    norm = BiasFreeLayerNorm(16)
    y = norm(x)

    assert y.shape == x.shape
    # Bias-free: normalizes by RMS only, weight starts at 1 -> unit-ish variance.
    assert torch.allclose(y.var(-1, unbiased=False).mean(), torch.tensor(1.0), atol=0.05)


def test_with_bias_layer_norm_shape_and_stats():
    x = torch.randn(2, 100, 16) * 5 + 3

    norm = WithBiasLayerNorm(16)
    y = norm(x)

    assert y.shape == x.shape
    assert torch.allclose(y.mean(-1), torch.zeros(2, 100), atol=1e-4)
    assert torch.allclose(y.var(-1, unbiased=False).mean(), torch.tensor(1.0), atol=0.05)


def test_layer_norm_2d_wrapper_with_bias():
    x = torch.randn(2, 48, 16, 16)

    ln = LayerNorm(48, "WithBias")
    y = ln(x)

    assert y.shape == x.shape


def test_layer_norm_2d_wrapper_bias_free():
    x = torch.randn(2, 48, 16, 16)

    ln = LayerNorm(48, "BiasFree")
    y = ln(x)

    assert y.shape == x.shape


def test_layer_norm_realistic_resolution():
    x = torch.randn(1, 48, 256, 256)

    ln = LayerNorm(48)
    y = ln(x)

    assert y.shape == x.shape


if __name__ == "__main__":
    test_to_3d_to_4d_roundtrip()
    test_bias_free_layer_norm_shape_and_stats()
    test_with_bias_layer_norm_shape_and_stats()
    test_layer_norm_2d_wrapper_with_bias()
    test_layer_norm_2d_wrapper_bias_free()
    test_layer_norm_realistic_resolution()
    print("All layers tests passed.")
