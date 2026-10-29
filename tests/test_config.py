"""
Config schema tests.

These guard the contract between configs/srrt_base.yaml and the code that
reads it, so a renamed or dropped key fails loudly here instead of part-way
through a long training run.
"""

import pytest
import yaml

CONFIG_PATH = "configs/srrt_base.yaml"


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def test_top_level_sections_present(cfg):
    expected = {
        "experiment", "dataset", "training", "optimizer", "scheduler",
        "model", "loss", "evaluation", "logging", "checkpointing", "device",
    }
    assert expected <= set(cfg)


def test_experiment_section(cfg):
    exp = cfg["experiment"]
    assert isinstance(exp["name"], str) and exp["name"]
    assert isinstance(exp["seed"], int)
    for key in ("save_dir", "log_dir", "output_dir", "results_dir"):
        assert isinstance(exp[key], str)


def test_dataset_section(cfg):
    ds = cfg["dataset"]
    assert ds["name"] == "gopro"
    assert ds["root"] == "data/Gopro"

    for split in ("train", "val"):
        s = ds[split]
        assert s["crop_size"] % 8 == 0, "crop_size must be divisible by 8 (3 downsampling stages)"
        assert s["crop_mode"] in ("random", "center", "none")
        assert isinstance(s["batch_size"], int) and s["batch_size"] > 0
        assert isinstance(s["num_workers"], int) and s["num_workers"] >= 0

    # Training must use random crops + augmentation; validation must be
    # deterministic, or val PSNR is not comparable across epochs.
    assert ds["train"]["crop_mode"] == "random" and ds["train"]["augment"] is True
    assert ds["val"]["crop_mode"] in ("center", "none") and ds["val"]["augment"] is False


def test_restormer_architecture_matches_paper_defaults(cfg):
    r = cfg["model"]["restormer"]

    assert r["dim"] == 48
    assert r["encoder_blocks"] == [4, 6, 6, 8]
    assert r["decoder_blocks"] == [6, 6, 4]
    assert r["num_heads"] == [1, 2, 4, 8]
    assert r["refinement_blocks"] == 4
    assert r["expansion_factor"] == 2.66
    assert r["layer_norm_type"] in ("WithBias", "BiasFree")

    # 3 encoder levels + bottleneck; heads must line up with them.
    assert len(r["encoder_blocks"]) == len(r["num_heads"]) == 4
    assert len(r["decoder_blocks"]) == 3

    # Every level's width must divide evenly into its head count.
    for level, heads in enumerate(r["num_heads"]):
        assert (r["dim"] * 2 ** level) % heads == 0


def test_latent_dim_consistent_with_restormer_dim(cfg):
    assert cfg["model"]["latent_dim"] == cfg["model"]["restormer"]["dim"] * 8


def test_all_srrt_modules_disabled_for_baseline(cfg):
    """The shipped default is the pure Restormer baseline (ablation A)."""
    modules = cfg["model"]["modules"]
    assert set(modules) == {"seecoder", "diffae", "lcr", "rectified_flow", "sra"}
    assert all(m["enabled"] is False for m in modules.values())


def test_optimizer_and_scheduler(cfg):
    opt = cfg["optimizer"]
    assert opt["type"] == "AdamW"
    assert opt["learning_rate"] == 0.0002
    assert opt["weight_decay"] == 0.0001
    assert opt["betas"] == [0.9, 0.999]

    assert cfg["scheduler"]["type"] == "CosineAnnealingLR"
    assert cfg["scheduler"]["T_max"] == cfg["training"]["epochs"], \
        "scheduler T_max should match training.epochs"


def test_loss_weights_and_perceptual_choice(cfg):
    loss = cfg["loss"]
    for key in ("l1_weight", "perceptual_weight", "flow_weight", "consistency_weight"):
        assert isinstance(loss[key], (int, float)) and loss[key] >= 0

    # LPIPS is the reported evaluation metric, so training must not optimize it.
    assert loss["perceptual_type"] == "vgg19"
    assert "LPIPS" in cfg["evaluation"]["metrics"]


def test_training_and_device_safety_settings(cfg):
    tr = cfg["training"]
    assert tr["grad_accum_steps"] >= 1
    assert tr["grad_clip_norm"] >= 0

    dev = cfg["device"]
    assert dev["mixed_precision"] is True
    assert 0 < dev["max_memory_fraction"] <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
