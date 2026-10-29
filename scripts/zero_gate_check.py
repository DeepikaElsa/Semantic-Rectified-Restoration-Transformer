"""
Phase 7 -- zero-gate equivalence check.

Two properties the whole ablation study rests on:

  A. SRRT with all modules DISABLED == Restormer            (exact)
  B. SRRT with all modules ENABLED, BEFORE training ~= Restormer
     (because every module joins through a zero-initialized gate)

If B did not hold, enabling a module would perturb the network at
initialization and any measured ablation difference would be confounded with
that perturbation rather than reflecting what the module learned.

Reports max absolute error, mean absolute error and relative error.

    python scripts/zero_gate_check.py
    python scripts/zero_gate_check.py --size 256 --json results/zero_gate.json
"""

import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbone.restormer import Restormer  # noqa: E402
from models.srrt import SRRT  # noqa: E402
from utils.ablation import ABLATIONS, apply_ablation  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def errors(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a - b).abs()
    denom = b.abs().mean().clamp(min=1e-12)
    return {
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "relative_error": float(diff.mean() / denom),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/srrt_base.yaml")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 3, args.size, args.size, device=device)

    print("=" * 74)
    print(f"ZERO-GATE CHECK  (input 1x3x{args.size}x{args.size}, device={device.type}, "
          f"seed={args.seed}, float32)")
    print("=" * 74)

    # Reference: the bare Restormer backbone.
    set_seed(args.seed)
    ref_srrt = SRRT.from_config(apply_ablation(base_cfg, "baseline_restormer")).to(device).eval()
    restormer = Restormer.from_config(base_cfg).to(device).eval()
    restormer.load_state_dict(ref_srrt.backbone.state_dict())

    with torch.no_grad():
        y_ref = restormer(x)

    report: dict[str, dict] = {}
    all_ok = True

    for name in ABLATIONS:
        cfg = apply_ablation(base_cfg, name)

        # Same seed => the shared backbone weights are identical across
        # configurations, so any output difference comes only from the modules.
        set_seed(args.seed)
        model = SRRT.from_config(cfg).to(device).eval()
        model.backbone.load_state_dict(restormer.state_dict())

        with torch.no_grad():
            y = model(x)["output"]

        e = errors(y, y_ref)
        exact = name == "baseline_restormer"
        # float32 forward through ~44 transformer blocks: exact equality only
        # expected when the module path is entirely absent.
        tol = 0.0 if exact else 1e-5
        ok = e["max_abs_error"] <= tol if exact else e["max_abs_error"] < tol
        all_ok &= ok

        report[name] = {**e, "exact_required": exact, "passed": bool(ok),
                        "params": sum(p.numel() for p in model.parameters())}

        label = "EXACT" if exact else "~=   "
        print(f"[{'PASS' if ok else 'FAIL'}] {label} {name:<26} "
              f"max={e['max_abs_error']:.3e}  mean={e['mean_abs_error']:.3e}  "
              f"rel={e['relative_error']:.3e}")

    # Confirm the gates really are zero at init (the mechanism behind the above).
    print("-" * 74)
    set_seed(args.seed)
    full = SRRT.from_config(apply_ablation(base_cfg, "full_srrt")).to(device)
    gates = {
        "srrt.gamma_latent": float(full.gamma_latent.abs().max()),
        "sra.gamma": float(full.sra.gamma.abs().max()),
        "flow.proj_out.weight": float(full.flow.proj_out.weight.abs().max()),
        "flow.proj_out.bias": float(full.flow.proj_out.bias.abs().max()),
    }
    gates_ok = all(v == 0.0 for v in gates.values())
    all_ok &= gates_ok
    print(f"[{'PASS' if gates_ok else 'FAIL'}] zero-initialized gates at init: "
          + ", ".join(f"{k}={v:g}" for k, v in gates.items()))

    # And that the flow velocity field is identically zero at init.
    z = torch.randn(2, full.latent_dim, 8, 8, device=device)
    with torch.no_grad():
        v = full.flow.velocity(z, torch.rand(2, device=device))
    v_ok = float(v.abs().max()) == 0.0
    all_ok &= v_ok
    print(f"[{'PASS' if v_ok else 'FAIL'}] rectified-flow velocity is exactly 0 at init "
          f"(max|v|={float(v.abs().max()):g}) -> ODE is the identity map")

    print("=" * 74)
    print("RESULT: " + ("all zero-gate properties hold" if all_ok else "ZERO-GATE PROPERTY VIOLATED"))
    print("=" * 74)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"input_size": args.size, "seed": args.seed,
                       "device": device.type, "per_ablation": report,
                       "gates": gates, "velocity_max_at_init": float(v.abs().max()),
                       "all_passed": bool(all_ok)}, f, indent=2)
        print(f"Written to {args.json}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
