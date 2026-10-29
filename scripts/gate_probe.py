"""
Read-only per-epoch gate logger for a live training run.

    python scripts/gate_probe.py --experiment seecoder_diffae_lcr_flow_detach_diag

Watches `logs/<exp>/train_log.csv` and, each time a new epoch row appears,
reads `checkpoints/<exp>/latest.pth` and records the gate magnitudes. Writes to
`logs/<exp>/gate_log.csv`.

Touches nothing the trainer owns: it opens the checkpoint read-only and never
writes into the training log. A torn read (the trainer rewriting latest.pth
mid-load) is caught and retried on the next poll rather than being reported as
a value.

Exists because train.py does not log gate magnitudes and must not be modified
while a diagnostic is in flight.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import torch

FIELDS = ["epoch", "gamma_latent", "sra_gamma", "flow_proj_out_w", "inject_w_norm"]


def gates(path: str) -> dict[str, float] | None:
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None                      # torn read; try again next poll
    sd = ck.get("model_state", {})

    def val(key, fn=lambda t: float(t.abs().max())):
        t = sd.get(key)
        return fn(t) if t is not None else float("nan")

    return {
        "epoch": ck.get("epoch"),
        "gamma_latent": val("gamma_latent"),
        "sra_gamma": val("sra.gamma"),
        "flow_proj_out_w": val("flow.proj_out.weight"),
        "inject_w_norm": val("inject.weight", lambda t: float(t.norm())),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--poll-seconds", type=float, default=20.0)
    ap.add_argument("--max-hours", type=float, default=6.0)
    args = ap.parse_args()

    log = os.path.join("logs", args.experiment, "train_log.csv")
    ckpt = os.path.join("checkpoints", args.experiment, "latest.pth")
    out = os.path.join("logs", args.experiment, "gate_log.csv")

    seen: set[int] = set()
    if os.path.exists(out):
        with open(out, newline="") as f:
            seen = {int(r["epoch"]) for r in csv.DictReader(f) if r.get("epoch")}
    else:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()

    print(f"gate_probe: watching {log} -> {out}")
    started = time.time()

    while (time.time() - started) / 3600.0 < args.max_hours:
        time.sleep(args.poll_seconds)
        if not (os.path.exists(log) and os.path.exists(ckpt)):
            continue
        try:
            with open(log, newline="") as f:
                epochs = {int(r["epoch"]) for r in csv.DictReader(f) if r.get("epoch")}
        except (OSError, ValueError):
            continue

        for e in sorted(epochs - seen):
            g = gates(ckpt)
            # Only record when the checkpoint has caught up with the log row.
            if g is None or g.get("epoch") != e:
                continue
            with open(out, "a", newline="") as f:
                csv.DictWriter(f, FIELDS).writerow(g)
            seen.add(e)
            print(f"  epoch {e}: gamma_latent={g['gamma_latent']:.6g} "
                  f"inject_w={g['inject_w_norm']:.6g}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
