"""Reproducibility helpers: seeding and environment capture (spec section 23)."""

import os
import platform
import random
import subprocess
import sys
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy and PyTorch (CPU + all CUDA devices).

    Args:
        seed: base seed.
        deterministic: if True, force deterministic cuDNN algorithms and
            disable benchmark autotuning. This makes runs bit-reproducible at
            a real speed cost, so it is off by default and driven by
            `experiment.deterministic` in the YAML config.

    Note: DataLoader workers are seeded by PyTorch from the base torch seed,
    which is why data/transforms.py draws randomness from torch rather than
    the `random` module.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: some ops have no deterministic kernel; warn instead of
        # crashing a long training run.
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def _git_hash() -> str | None:
    """Current commit hash, or None if this is not a git repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def collect_environment() -> dict[str, Any]:
    """
    Snapshot the runtime environment, stored alongside every checkpoint and
    results summary so any reported number can be traced to the exact setup
    that produced it.
    """
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": torch.cuda.is_available(),
        "git_hash": _git_hash(),
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_total_memory_mb"] = props.total_memory // (1024 ** 2)
        info["gpu_capability"] = f"{props.major}.{props.minor}"

    return info
