"""
Code-freeze audit: every ablation configuration, end to end, on CPU.

Covers audit sections 1-4:

    1  pipeline      construction, forward, output shape, finiteness,
                     backward, finite gradients, parameter updates, device
    2  architecture  Restormer-only == pure Restormer (bitwise);
                     zero-initialized gates
    3  ablation      exactly the intended modules enabled, as a machine-readable
                     table checked against the expected matrix
    4  loss          the training objective carries the terms the enabled
                     modules are supposed to contribute

    python scripts/audit_code_freeze.py
    python scripts/audit_code_freeze.py --json research/tables/code_freeze_audit.json

RUNS ON CPU BY DEFAULT and uses a deliberately tiny architecture, so it is safe
to run beside a live GPU training job. Pass --device cuda only when the GPU is
free. Exits non-zero if any check fails, so it can gate a freeze.

The backward pass uses the FULL training objective -- L1 plus the flow and LCR
terms when those modules are active -- not just the pixel term. Scoring only the
pixel loss would report a module as disconnected when train.py does train it,
and would equally hide a module whose auxiliary loss is silently dropped.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses.combined_loss import CombinedLoss  # noqa: E402
from models.backbone.restormer import Restormer  # noqa: E402
from models.srrt import SRRT  # noqa: E402
from utils.ablation import ABLATIONS, apply_ablation  # noqa: E402
from utils.seed import set_seed  # noqa: E402

MODULES = ["seecoder", "diffae", "lcr", "rectified_flow", "sra"]

# Section 3: the intended enable matrix. This is the ground truth the ladder
# rests on -- if a row is wrong, every downstream comparison is confounded.
EXPECTED = {
    "baseline_restormer":       [False, False, False, False, False],
    "seecoder":                 [True,  False, False, False, False],
    "seecoder_diffae":          [True,  True,  False, False, False],
    "seecoder_diffae_lcr":      [True,  True,  True,  False, False],
    "seecoder_diffae_lcr_flow": [True,  True,  True,  True,  False],
    "full_srrt":                [True,  True,  True,  True,  True],
}

# Which auxiliary loss each module is expected to contribute.
AUX_FOR = {"rectified_flow": "flow_loss", "lcr": "lcr_loss"}

# module_summary() name -> attribute prefix in named_parameters()
ATTR_OF = {"rectified_flow": "flow"}


def tiny_cfg(path: str) -> dict:
    """Small architecture: same shapes and logic, far less compute."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    r = cfg["model"]["restormer"]
    r["dim"] = 16
    r["encoder_blocks"] = [1, 1, 1, 1]
    r["decoder_blocks"] = [1, 1, 1]
    r["num_heads"] = [1, 1, 1, 1]
    r["refinement_blocks"] = 1
    cfg["model"]["latent_dim"] = 16 * 8
    m = cfg["model"]["modules"]
    m["seecoder"].update({"embed_dim": 32, "dim": 8, "num_heads": 1, "blocks_per_stage": 1})
    m["diffae"].update({"latent_dim": 32, "dim": 8})
    m["lcr"].update({"hidden_dim": 64, "proj_dim": 32, "num_heads": 2, "refinement_blocks": 1})
    m["rectified_flow"].update({"hidden_dim": 32, "time_embed_dim": 32,
                                "num_blocks": 1, "steps": 4})
    m["sra"].update({"num_heads": 2})
    return cfg


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, bool, str]] = []

    def add(self, group: str, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((group, name, bool(ok), detail))
        return bool(ok)

    @property
    def failed(self) -> list[tuple[str, str, bool, str]]:
        return [r for r in self.rows if not r[2]]

    def report(self) -> None:
        cur = None
        for group, name, ok, detail in self.rows:
            if group != cur:
                print(f"\n--- {group} " + "-" * max(0, 66 - len(group)))
                cur = group
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:<46} {detail}")


def audit_one(name: str, base_cfg: dict, device: torch.device,
              size: int, chk: Checks, x_ref: torch.Tensor, ref_out: torch.Tensor,
              ref_state: dict) -> dict:
    cfg = apply_ablation(copy.deepcopy(base_cfg), name)
    rec: dict = {"ablation": name}

    # ---- 1. construction ------------------------------------------------
    set_seed(42)
    try:
        model = SRRT.from_config(cfg).to(device)
    except Exception as exc:
        chk.add(name, "model constructs", False, repr(exc))
        return rec
    chk.add(name, "model constructs", True)

    summary = model.module_summary()
    enabled = [bool(summary[m]["enabled"]) for m in MODULES]
    rec["enabled"] = dict(zip(MODULES, enabled))
    rec["params"] = int(summary["total"])

    # ---- 3. ablation integrity -----------------------------------------
    chk.add(name, "module flags match expected matrix", enabled == EXPECTED[name],
            f"{enabled} vs {EXPECTED[name]}")

    # ---- device placement ----------------------------------------------
    devs = {p.device.type for p in model.parameters()}
    chk.add(name, "all parameters on target device", devs == {device.type},
            f"{sorted(devs)}")

    x = torch.randn(1, 3, size, size, device=device)
    y = torch.randn(1, 3, size, size, device=device)

    # ---- 1. forward ------------------------------------------------------
    model.eval()
    with torch.no_grad():
        out = model(x)
    pred = out["output"]
    chk.add(name, "output shape == input shape", tuple(pred.shape) == tuple(x.shape),
            f"{tuple(pred.shape)}")
    chk.add(name, "forward output finite", bool(torch.isfinite(pred).all()))

    # ---- 2. zero-gate: identical to pure Restormer at init ---------------
    # Must use the SAME input the reference output was computed from; comparing
    # across two different random inputs makes even the baseline row fail.
    model.backbone.load_state_dict(ref_state)
    with torch.no_grad():
        gated = model(x_ref)["output"]
    max_abs = float((gated - ref_out).abs().max())
    rec["zero_gate_max_abs_error"] = max_abs
    chk.add(name, "== pure Restormer at init (bitwise)", max_abs == 0.0,
            f"max|d|={max_abs:.3e}")

    for gate_name, tensor in (("gamma_latent", model.gamma_latent),
                              ("sra.gamma", getattr(model.sra, "gamma", None) if model.sra else None),
                              ("flow.proj_out.weight",
                               model.flow.proj_out.weight if model.flow else None)):
        if tensor is None:
            continue
        v = float(tensor.detach().abs().max())
        rec[f"gate_{gate_name}"] = v
        chk.add(name, f"gate {gate_name} == 0 at init", v == 0.0, f"{v:g}")

    # ---- 4. loss integrity ----------------------------------------------
    model.train()
    out_t = model(x, target=y)
    aux_present = {k: (out_t.get(v) is not None) for k, v in AUX_FOR.items()}
    rec["aux_losses"] = aux_present
    for mod, key in AUX_FOR.items():
        want = bool(summary[mod]["enabled"])
        chk.add(name, f"{key} present iff {mod} enabled",
                aux_present[mod] == want, f"present={aux_present[mod]} enabled={want}")

    # Weights only, no perceptual net: keeps the audit light and lets the
    # composition be checked by exact arithmetic.
    loss_fn = CombinedLoss(l1_weight=1.0, perceptual_weight=0.1, flow_weight=0.1,
                           consistency_weight=0.1, use_perceptual=False).to(device)
    total, comps = loss_fn(out_t["output"], y,
                           flow_loss=out_t.get("flow_loss"),
                           lcr_loss=out_t.get("lcr_loss"))

    expect = 1.0 * comps["l1"]
    if "flow" in comps:
        expect = expect + 0.1 * comps["flow"]
    if "lcr" in comps:
        expect = expect + 0.1 * comps["lcr"]
    chk.add(name, "total == sum of weighted components",
            torch.allclose(total, expect, atol=1e-6),
            f"total={float(total):.6f}")
    chk.add(name, "loss finite", bool(torch.isfinite(total)))
    rec["loss_components"] = sorted(k for k in comps if k != "total")

    # ---- 1. backward + updates ------------------------------------------
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt.zero_grad(set_to_none=True)
    total.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    chk.add(name, "gradients exist", bool(grads), f"{len(grads)} tensors")
    chk.add(name, "all gradients finite",
            all(bool(torch.isfinite(g).all()) for g in grads))
    opt.step()

    changed = sum(1 for n, p in model.named_parameters()
                  if not torch.equal(p.detach(), before[n]))
    chk.add(name, "optimizer updates parameters", changed > 0,
            f"{changed}/{len(before)} tensors changed")

    # ---- 2/3. connectivity: enabled modules must receive gradient --------
    # Two extra steps so the zero gate opens; on step 1 only the gate itself
    # has a gradient, which is correct behaviour, not disconnection.
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        o = model(x, target=y)
        t, _ = loss_fn(o["output"], y, flow_loss=o.get("flow_loss"),
                       lcr_loss=o.get("lcr_loss"))
        t.backward()
        opt.step()

    opt.zero_grad(set_to_none=True)
    o = model(x, target=y)
    t, _ = loss_fn(o["output"], y, flow_loss=o.get("flow_loss"),
                   lcr_loss=o.get("lcr_loss"))
    t.backward()

    live, untrained = {}, {}
    for mod in MODULES:
        if not summary[mod]["enabled"]:
            continue
        prefix = ATTR_OF.get(mod, mod)
        named = [(n, p) for n, p in model.named_parameters() if n.startswith(prefix)]
        dead = [n for n, p in named
                if p.grad is None or float(p.grad.abs().sum()) == 0]
        n_live = len(named) - len(dead)
        live[mod] = f"{n_live}/{len(named)}"
        chk.add(name, f"module '{mod}' receives gradient", n_live > 0, live[mod])
        if dead:
            # Reported, not failed: a sub-block can legitimately sit outside the
            # objective (DiffAE's semantic_head feeds visualization only). It is
            # surfaced so it cannot grow unnoticed into the parameter counts.
            untrained[mod] = dead
    rec["connectivity"] = live
    rec["untrained_tensors"] = untrained
    if untrained:
        flat = [n for v in untrained.values() for n in v]
        chk.add(name, "untrained tensors (reported, not a failure)", True,
                f"{len(flat)}: {', '.join(flat[:4])}"
                + (" ..." if len(flat) > 4 else ""))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", default="configs/srrt_base.yaml")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cpu by default so this is safe beside a live training job")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--json", default="research/tables/code_freeze_audit.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    base_cfg = tiny_cfg(args.config)
    chk = Checks()

    print("=" * 78)
    print(f"CODE-FREEZE AUDIT  (device={device.type}, input 1x3x{args.size}x{args.size}, "
          f"tiny architecture)")
    print("=" * 78)

    # Reference: the bare Restormer backbone, weights shared with every SRRT.
    set_seed(42)
    ref_srrt = SRRT.from_config(
        apply_ablation(copy.deepcopy(base_cfg), "baseline_restormer")).to(device).eval()
    restormer = Restormer.from_config(base_cfg).to(device).eval()
    restormer.load_state_dict(ref_srrt.backbone.state_dict())
    ref_state = {k: v.clone() for k, v in restormer.state_dict().items()}
    x_ref = torch.randn(1, 3, args.size, args.size, device=device)
    with torch.no_grad():
        ref_out = restormer(x_ref)
    torch.manual_seed(0)

    records = []
    for name in EXPECTED:
        if name not in ABLATIONS:
            chk.add("registry", f"{name} exists in utils.ablation", False)
            continue
        # Re-draw the same input each time so the zero-gate comparison is valid.
        torch.manual_seed(1234)
        records.append(audit_one(name, base_cfg, device, args.size, chk,
                                 x_ref, ref_out, ref_state))

    chk.add("registry", "no unexpected ablations defined",
            set(ABLATIONS) == set(EXPECTED),
            f"{sorted(set(ABLATIONS) ^ set(EXPECTED))}")

    chk.report()

    # Section 3 machine-readable table.
    print("\n" + "=" * 78)
    print("ABLATION ENABLE MATRIX")
    print("=" * 78)
    head = f"{'Configuration':<26}" + "".join(f"{m:>16}" for m in MODULES)
    print(head)
    print("-" * len(head))
    for r in records:
        if "enabled" not in r:
            continue
        print(f"{r['ablation']:<26}" +
              "".join(f"{str(r['enabled'][m]):>16}" for m in MODULES))

    n_pass = len(chk.rows) - len(chk.failed)
    print("\n" + "=" * 78)
    print(f"RESULT: {n_pass}/{len(chk.rows)} checks passed")
    for group, name, _, detail in chk.failed:
        print(f"  FAIL  {group}: {name}  {detail}")
    print("=" * 78)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({
                "device": device.type,
                "input_size": args.size,
                "architecture": "tiny (audit only, not the trained config)",
                "expected_matrix": {k: dict(zip(MODULES, v)) for k, v in EXPECTED.items()},
                "records": records,
                "checks": [{"group": g, "name": n, "passed": p, "detail": d}
                           for g, n, p, d in chk.rows],
                "checks_passed": n_pass,
                "checks_total": len(chk.rows),
                "all_passed": not chk.failed,
            }, f, indent=2)
        print(f"wrote {args.json}")

    return 0 if not chk.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
