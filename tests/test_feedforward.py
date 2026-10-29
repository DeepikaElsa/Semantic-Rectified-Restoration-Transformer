import torch

from models.common.feedforward import GDFN


def test_gdfn_shape_small():
    x = torch.randn(2, 48, 16, 16)

    model = GDFN(dim=48)
    y = model(x)

    assert y.shape == x.shape


def test_gdfn_hidden_dim_matches_expansion_factor():
    dim, expansion_factor = 48, 2.66
    expected_hidden = int(dim * expansion_factor)

    model = GDFN(dim=dim, expansion_factor=expansion_factor)

    # project_in outputs 2x hidden_features (split into gate + value branches)
    assert model.project_in.out_channels == expected_hidden * 2
    assert model.project_out.in_channels == expected_hidden


def test_gdfn_realistic_resolution():
    x = torch.randn(1, 48, 256, 256)

    model = GDFN(dim=48)
    y = model(x)

    assert y.shape == x.shape


if __name__ == "__main__":
    test_gdfn_shape_small()
    test_gdfn_hidden_dim_matches_expansion_factor()
    test_gdfn_realistic_resolution()
    print("All feedforward tests passed.")
