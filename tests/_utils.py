"""
Shared helpers for resource-aware tests (CPU RAM / GPU VRAM / time).
Not a test module itself (no test_ prefix), imported by the tests/ suite.
"""

import gc
import time
from contextlib import contextmanager

import psutil
import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextmanager
def measure_resources(device):
    """
    Measures wall-clock time, CPU RSS delta (MB) and, on CUDA, peak
    GPU memory allocated (MB) for the wrapped block. Results are
    written into the yielded dict after the block exits.
    """

    gc.collect()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    process = psutil.Process()
    cpu_before = process.memory_info().rss / (1024 ** 2)

    stats = {}
    start = time.perf_counter()

    yield stats

    if device.type == "cuda":
        torch.cuda.synchronize()
        stats["gpu_peak_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        stats["gpu_peak_mb"] = 0.0

    stats["elapsed_s"] = time.perf_counter() - start

    cpu_after = process.memory_info().rss / (1024 ** 2)
    stats["cpu_delta_mb"] = cpu_after - cpu_before


@contextmanager
def assert_no_quadratic_attention(max_allowed_dim):
    """
    Wraps torch.matmul to record the shape of every matmul performed
    inside the block, then asserts none of them produced a square
    matrix larger than `max_allowed_dim` on its last two dims.

    This directly guards against the HW x HW attention regression:
    at H=W=256 an accidental spatial-attention matmul produces a
    65536 x 65536 matrix, which this catches immediately regardless
    of whether the allocation happens to succeed.
    """

    recorded_shapes = []
    original_matmul = torch.matmul

    def patched_matmul(a, b, *args, **kwargs):
        out = original_matmul(a, b, *args, **kwargs)
        recorded_shapes.append(tuple(out.shape))
        return out

    torch.matmul = patched_matmul
    try:
        yield recorded_shapes
    finally:
        torch.matmul = original_matmul

    for shape in recorded_shapes:
        if len(shape) >= 2:
            last, second_last = shape[-1], shape[-2]
            assert not (last > max_allowed_dim and second_last > max_allowed_dim), (
                f"Found a matmul output of shape {shape}, which exceeds the "
                f"allowed {max_allowed_dim}x{max_allowed_dim} bound — this "
                f"looks like an HW x HW spatial attention matrix, not "
                f"channel/transposed attention."
            )
