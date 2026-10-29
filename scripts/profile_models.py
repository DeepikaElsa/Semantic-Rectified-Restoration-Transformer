"""
Computational cost of every ablation configuration (Phase 10).

SRRT adds modules on top of Restormer, so any accuracy gain has to be read
against what it costs. This measures, per configuration:

    total parameters
    trainable parameters
    model size on disk (fp32 state_dict bytes)
    forward FLOPs (if `thop` or `fvcore` is installed; otherwise reported unmeasured)
    inference latency, mean and std over timed repetitions
    peak GPU memory during inference

    python scripts/profile_models.py
    python scripts/profile_models.py --resolution 256 --repeats 50 --out research/tables/profile.json

METHODOLOGY (identical for every configuration, which is the entire point)
--------------------------------------------------------------------------
* input          : batch 1, 3 x R x R, R = --resolution (default 256, the
                   validation crop size), random uniform in [0,1]
* precision      : fp32 by default; --amp measures autocast instead
* warm-up        : --warmup forward passes, discarded. The first CUDA call pays
                   context creation, cuDNN autotuning and lazy kernel loading;
                   including it inflates the first configuration measured and
                   makes ordering an artifact of measurement order.
* timing         : torch.cuda.synchronize() before and after EVERY repetition.
                   CUDA is asynchronous, so timing without synchronizing measures
                   how fast Python can enqueue work, not how fast the GPU runs it.
* repetitions    : --repeats, reported as mean +/- std
* memory         : torch.cuda.reset_peak_memory_stats() before, max_memory_allocated() after
* weights        : randomly initialized. Cost depends on architecture, not on
                   learned values, so no checkpoint is needed and none is loaded.
                   Metrics are NEVER produced here -- accuracy comes from
                   evaluate.py alone.
* isolation      : each configuration is built, measured, and freed before the
                   next, with the CUDA cache emptied between, so one model's
                   allocations cannot inflate the next model's peak.
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.srrt import SRRT  # noqa: E402
from utils.ablation import ABLATIONS, apply_ablation  # noqa: E402
from utils.research import ABLATION_ORDER, base_provenance  # noqa: E402


def count_flops(model, x):
    """
    Forward FLOPs, or (None, reason) when no counter is installed.

    Returns MACs-derived GFLOPs. Both backends count multiply-accumulates; the
    convention here is GFLOPs = 2 * MACs, stated explicitly because published
    numbers differ by exactly a factor of two depending on this choice.
    """
    try:
        from thop import profile as thop_profile
        macs, _ = thop_profile(model, inputs=(x,), verbose=False)
        return 2.0 * macs / 1e9, "thop (GFLOPs = 2 x MACs)"
    except ImportError:
        pass
    except Exception as e:
        return None, f"thop failed: {e}"

    try:
        from fvcore.nn import FlopCountAnalysis
        macs = FlopCountAnalysis(model, x).unsupported_ops_warnings(False).total()
        return 2.0 * macs / 1e9, "fvcore (GFLOPs = 2 x MACs)"
    except ImportError:
        return None, "not measured: install `thop` or `fvcore`"
    except Exception as e:
        return None, f"fvcore failed: {e}"


def gpu_name(device) -> str | None:
    """GPU name, or None. Never fatal: the profile must survive an odd CUDA state."""
    if device.type != "cuda":
        return None
    try:
        return torch.cuda.get_device_name(device.index or 0)
    except Exception:
        return None


def state_dict_bytes(model) -> int:
    return sum(t.numel() * t.element_size() for t in model.state_dict().values())


def profile_one(name: str, cfg: dict, device, resolution: int, warmup: int,
                repeats: int, use_amp: bool, want_flops: bool) -> dict:
    cfg = apply_ablation(cfg, name)
    model = SRRT.from_config(cfg).to(device).eval()

    summary = model.module_summary()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    x = torch.rand(1, 3, resolution, resolution, device=device)

    gflops, flops_backend = (None, "skipped (--no-flops)")
    if want_flops:
        gflops, flops_backend = count_flops(model, x)

    autocast = torch.amp.autocast(device.type, enabled=use_amp)

    with torch.no_grad():
        for _ in range(warmup):
            with autocast:
                model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    times = []
    with torch.no_grad():
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with autocast:
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()      # CUDA is async; without this we
            times.append(time.perf_counter() - t0)   # would time the enqueue only

    peak = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
            if device.type == "cuda" else None)

    result = {
        "ablation": name,
        "modules_enabled": {k: summary[k]["enabled"] for k in
                            ("seecoder", "diffae", "lcr", "rectified_flow", "sra")},
        "module_params": {k: summary[k]["params"] for k in
                          ("seecoder", "diffae", "lcr", "rectified_flow", "sra", "backbone")},
        "params_total": total,
        "params_trainable": trainable,
        "model_size_mb": state_dict_bytes(model) / 1024 ** 2,
        "gflops": gflops,
        "flops_backend": flops_backend,
        "inference_s": statistics.mean(times),
        "inference_s_std": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "inference_s_min": min(times),
        "peak_vram_mb": peak,
    }

    # Free before the next configuration so its peak VRAM is its own.
    del model, x
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", default="configs/srrt_base.yaml")
    ap.add_argument("--ablations", default=",".join(ABLATION_ORDER),
                    help=f"comma-separated subset of {sorted(ABLATIONS)}")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--amp", action="store_true", help="measure under autocast")
    ap.add_argument("--no-flops", action="store_true")
    ap.add_argument("--out", default="research/tables/profile.json")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="force the device. Use cpu to profile without competing "
                         "for VRAM with a training run in progress.")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available.")
    with open(args.config, "r") as f:
        base_cfg = yaml.safe_load(f)

    names = [n.strip() for n in args.ablations.split(",") if n.strip()]
    unknown = [n for n in names if n not in ABLATIONS]
    if unknown:
        raise SystemExit(f"unknown ablation(s) {unknown}; available: {sorted(ABLATIONS)}")

    print(f"device={device}  resolution={args.resolution}  batch=1  "
          f"warmup={args.warmup}  repeats={args.repeats}  amp={args.amp}")
    print("-" * 96)
    print(f"{'configuration':<28}{'params':>12}{'GFLOPs':>10}{'ms/img':>12}"
          f"{'+/- ms':>9}{'VRAM MB':>10}{'size MB':>10}")
    print("-" * 96)

    models = []
    for name in names:
        r = profile_one(name, base_cfg, device, args.resolution, args.warmup,
                        args.repeats, args.amp, not args.no_flops)
        models.append(r)
        gf = "n/a" if r["gflops"] is None else f"{r['gflops']:.1f}"
        print(f"{name:<28}{r['params_total']:>12,}{gf:>10}"
              f"{r['inference_s'] * 1e3:>12.2f}{r['inference_s_std'] * 1e3:>9.2f}"
              f"{(r['peak_vram_mb'] or 0):>10.0f}{r['model_size_mb']:>10.1f}")

    out = base_provenance(
        artifact="model_profile",
        methodology={
            "input": f"batch 1, 3 x {args.resolution} x {args.resolution}, uniform [0,1]",
            "precision": "autocast" if args.amp else "fp32",
            "warmup_iters": args.warmup,
            "timed_iters": args.repeats,
            "synchronization": "torch.cuda.synchronize() around every timed iteration",
            "weights": "randomly initialized; cost is architecture-dependent, not weight-dependent",
            "isolation": "each configuration freed and CUDA cache emptied before the next",
            "flops_convention": "GFLOPs = 2 x MACs",
        },
        device=str(device),
        gpu=gpu_name(device),
        torch=torch.__version__,
        models=models,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("-" * 96)
    print(f"wrote {args.out}")
    if any(m["gflops"] is None for m in models):
        print("FLOPs unmeasured for at least one model "
              "(`pip install thop` to enable); tables will show '--'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
