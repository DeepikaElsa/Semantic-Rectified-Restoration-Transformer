import torch

from models.common.attention import MDTA
from tests._utils import assert_no_quadratic_attention, measure_resources, get_device


def test_mdta_shape_small():
    x = torch.randn(2, 48, 16, 16)

    model = MDTA(dim=48, num_heads=4)
    y = model(x)

    assert y.shape == x.shape


def test_mdta_shape_various_head_counts():
    for num_heads in (1, 2, 4, 8):
        x = torch.randn(1, 32, 8, 8)
        model = MDTA(dim=32, num_heads=num_heads)
        y = model(x)
        assert y.shape == x.shape


def test_mdta_is_channel_attention_not_spatial():
    """
    Regression test for the HW x HW bug: at H=W=64 (4096 spatial
    positions) the attention matrix must be head_dim x head_dim
    (a few dozen), never anything close to 4096 x 4096.
    """

    dim, num_heads, H, W = 48, 4, 64, 64
    head_dim = dim // num_heads

    x = torch.randn(1, dim, H, W)
    model = MDTA(dim=dim, num_heads=num_heads)

    with assert_no_quadratic_attention(max_allowed_dim=head_dim) as shapes:
        model(x)

    # Sanity: we actually captured matmuls, and the attention matrix's
    # square dims equal head_dim exactly (not just "small").
    square_shapes = [s for s in shapes if len(s) == 4 and s[-1] == s[-2]]
    assert any(s[-1] == head_dim for s in square_shapes), (
        f"Expected an attention matrix with square dim {head_dim}, "
        f"got shapes {shapes}"
    )


def test_mdta_at_realistic_resolution_stays_cheap():
    """
    This is the exact configuration that previously triggered a ~34GB
    allocation (dim=48, H=W=256, the encoder's first stage). With
    correct channel attention this must complete using a few MB.
    """

    device = get_device()
    dim, num_heads, H, W = 48, 1, 256, 256

    x = torch.randn(1, dim, H, W, device=device)
    model = MDTA(dim=dim, num_heads=num_heads).to(device)

    with assert_no_quadratic_attention(max_allowed_dim=dim):
        with measure_resources(device) as stats, torch.no_grad():
            y = model(x)

    assert y.shape == x.shape
    print(f"[MDTA @ 256x256] time={stats['elapsed_s']:.3f}s "
          f"gpu_peak={stats['gpu_peak_mb']:.1f}MB cpu_delta={stats['cpu_delta_mb']:.1f}MB")

    # Channel attention at this size should need low hundreds of MB at most,
    # nowhere near the multi-GB territory of the old HW x HW bug.
    assert stats["gpu_peak_mb"] < 1024 or device.type == "cpu"


if __name__ == "__main__":
    test_mdta_shape_small()
    test_mdta_shape_various_head_counts()
    test_mdta_is_channel_attention_not_spatial()
    test_mdta_at_realistic_resolution_stays_cheap()
    print("All attention tests passed.")
