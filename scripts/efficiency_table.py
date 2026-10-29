"""
Computational-cost comparison between runs, from measured artifacts only.

    python scripts/efficiency_table.py \
        --run "Restormer=baseline_restormer=results/restormer_baseline" \
        --run "Restormer + SeeCoder=seecoder=results/ablation_seecoder"

Each --run is LABEL=EXPERIMENT[=RESULTS_DIR]; RESULTS_DIR defaults to
`results/<EXPERIMENT>`.

Sources, all read-only:

    parameters          results/<run>/summary.json  (module_summary), else
                        logs/<run>/run_metadata.json
    train time          logs/<run>/train_log.csv    sum of epoch_time_s
    median epoch time   logs/<run>/train_log.csv
    peak VRAM (train)   logs/<run>/train_log.csv    max peak_vram_mb
    peak RAM  (train)   logs/<run>/train_log.csv    max peak_ram_mb
    inference time      results/<run>/summary.json  mean_inference_s
    peak VRAM (eval)    results/<run>/summary.json  peak_vram_mb

GFLOPs
------
NOT estimated. FLOP counting needs `thop` or `fvcore`; when neither is importable
the cell reads "unavailable" and the reason is recorded in the JSON. A guessed
FLOP number would be indistinguishable in the table from a measured one, so this
script never writes one. To populate it later, install a counter and run
`scripts/profile_models.py` (which needs a GPU), then pass --profile.

STANDBY OUTLIERS
----------------
Wall-clock epochs longer than --outlier-seconds are reported separately and
excluded from the train-time total and the median. On this machine Windows
Modern Standby suspended two baseline epochs mid-run, recording 17,303 s and
5,515 s of sleep as "epoch time". Including them would inflate the baseline's
cost and understate the ablation's relative overhead.

Touches no GPU; safe to run while a job is training (an in-progress run is
reported with the epochs it has finished so far).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.research import read_train_log, resolve_train_log, to_float  # noqa: E402

NA = "unavailable"


def flop_counter_available() -> tuple[bool, str]:
    for mod in ("thop", "fvcore"):
        try:
            __import__(mod)
            return True, mod
        except ImportError:
            continue
    return False, "neither `thop` nor `fvcore` is installed"


def _find(run_dir: str, name: str) -> str | None:
    for c in (os.path.join(run_dir, name), os.path.join(run_dir, "metrics", name)):
        if os.path.isfile(c):
            return c
    return None


def collect(label: str, experiment: str, results_dir: str,
            outlier_s: float, profile: dict | None) -> dict:
    rec: dict = {"label": label, "experiment": experiment, "results_dir": results_dir,
                 "sources": {}, "notes": []}

    # ---- training log ----------------------------------------------------
    try:
        log = resolve_train_log(experiment)
        rows = read_train_log(log)
        rec["sources"]["train_log"] = log
    except FileNotFoundError:
        rows = []
        rec["notes"].append("no train_log.csv found")

    if rows:
        times = [to_float(r.get("epoch_time_s")) for r in rows]
        times = [t for t in times if t is not None]
        clean = [t for t in times if t <= outlier_s]
        outliers = [(int(to_float(r["epoch"])), to_float(r["epoch_time_s"]))
                    for r in rows
                    if to_float(r.get("epoch_time_s")) is not None
                    and to_float(r["epoch_time_s"]) > outlier_s]

        rec["epochs_completed"] = len(rows)
        rec["train_time_h"] = sum(clean) / 3600.0 if clean else None
        rec["median_epoch_s"] = st.median(clean) if clean else None
        rec["excluded_outlier_epochs"] = outliers
        if outliers:
            rec["notes"].append(
                f"{len(outliers)} epoch(s) >{outlier_s:.0f}s excluded as system "
                "standby, not compute: " + ", ".join(f"ep{e} ({t:.0f}s)" for e, t in outliers))

        vram = [to_float(r.get("peak_vram_mb")) for r in rows]
        ram = [to_float(r.get("peak_ram_mb")) for r in rows]
        rec["peak_vram_train_mb"] = max([v for v in vram if v is not None], default=None)
        rec["peak_ram_train_mb"] = max([v for v in ram if v is not None], default=None)

    # ---- evaluation summary ---------------------------------------------
    spath = _find(results_dir, "summary.json")
    if spath:
        with open(spath) as f:
            s = json.load(f)
        rec["sources"]["summary"] = spath
        rec["parameters"] = s.get("parameters")
        rec["inference_s"] = s.get("mean_inference_s")
        rec["peak_vram_eval_mb"] = s.get("peak_vram_mb")
        rec["eval_epoch"] = s.get("epoch")
        ms = s.get("module_summary") or {}
        rec["modules_enabled"] = sorted(
            k for k, v in ms.items()
            if isinstance(v, dict) and v.get("enabled") and k != "backbone")
        rec["module_params"] = {
            k: v.get("params") for k, v in ms.items()
            if isinstance(v, dict) and v.get("enabled") and k != "backbone"}
    else:
        rec["notes"].append(f"not evaluated yet (no summary.json under {results_dir})")

    # Parameters can still come from the run metadata before evaluation exists.
    if rec.get("parameters") is None:
        mpath = os.path.join("logs", experiment, "run_metadata.json")
        if os.path.isfile(mpath):
            with open(mpath) as f:
                md = json.load(f)
            ms = md.get("module_summary") or {}
            rec["parameters"] = ms.get("total")
            rec["sources"]["run_metadata"] = mpath
            rec["modules_enabled"] = rec.get("modules_enabled") or sorted(
                k for k, v in ms.items()
                if isinstance(v, dict) and v.get("enabled") and k != "backbone")

    # ---- FLOPs -----------------------------------------------------------
    rec["gflops"] = None
    if profile:
        entry = profile.get(experiment) or profile.get(rec["label"])
        if isinstance(entry, dict):
            g = entry.get("gflops") or entry.get("GFLOPs")
            if isinstance(g, (int, float)):
                rec["gflops"] = float(g)
                rec["sources"]["profile"] = "profile.json"
    return rec


def fmt(v, spec: str = "", scale: float = 1.0) -> str:
    if v is None:
        return NA
    try:
        return format(v * scale, spec) if spec else str(v)
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run", action="append", required=True,
                    help="LABEL=EXPERIMENT[=RESULTS_DIR] (repeatable)")
    ap.add_argument("--outlier-seconds", type=float, default=1500.0,
                    help="epochs longer than this are treated as system standby")
    ap.add_argument("--profile", default=None,
                    help="optional profile.json from scripts/profile_models.py (for GFLOPs)")
    ap.add_argument("--out", default="research/tables/ablation_seecoder_efficiency.json")
    ap.add_argument("--markdown", default="research/tables/ablation_seecoder_efficiency.md")
    args = ap.parse_args()

    profile = None
    if args.profile and os.path.isfile(args.profile):
        with open(args.profile) as f:
            profile = json.load(f)

    have_flops, flop_reason = flop_counter_available()

    records = []
    for spec in args.run:
        parts = spec.split("=", 2)
        if len(parts) < 2:
            print(f"--run needs LABEL=EXPERIMENT[=RESULTS_DIR], got {spec!r}")
            return 2
        label, experiment = parts[0], parts[1]
        results_dir = parts[2] if len(parts) > 2 else os.path.join("results", experiment)
        records.append(collect(label, experiment, results_dir,
                               args.outlier_seconds, profile))

    rows = [
        ("Parameters", lambda r: f"{r['parameters']:,}" if r.get("parameters") else NA),
        ("Train time (h)", lambda r: fmt(r.get("train_time_h"), ".2f")),
        ("Epochs completed", lambda r: fmt(r.get("epochs_completed"))),
        ("Median epoch time (min)", lambda r: fmt(r.get("median_epoch_s"), ".2f", 1 / 60)),
        ("Peak VRAM, train (MB)", lambda r: fmt(r.get("peak_vram_train_mb"), ".0f")),
        ("Peak RAM, train (MB)", lambda r: fmt(r.get("peak_ram_train_mb"), ".0f")),
        ("Inference time / image (s)", lambda r: fmt(r.get("inference_s"), ".4f")),
        ("Peak VRAM, eval (MB)", lambda r: fmt(r.get("peak_vram_eval_mb"), ".0f")),
        ("GFLOPs", lambda r: fmt(r.get("gflops"), ".2f")),
    ]

    labels = [r["label"] for r in records]
    width = max([len(n) for n, _ in rows] + [26])
    print("=" * 78)
    print("COMPUTATIONAL COST (measured values only)")
    print("=" * 78)
    print("  " + "Metric".ljust(width) + "".join(f"{l:>26}" for l in labels))
    for name, get in rows:
        print("  " + name.ljust(width) + "".join(f"{get(r):>26}" for r in records))

    if not have_flops:
        print(f"\nGFLOPs unavailable: {flop_reason}. Not estimated by design.")
    for r in records:
        for n in r["notes"]:
            print(f"note [{r['label']}]: {n}")

    report = {"runs": records,
              "gflops_available": have_flops,
              "gflops_reason": None if have_flops else flop_reason,
              "outlier_seconds": args.outlier_seconds,
              "caveat": ("Train time and median epoch time are wall-clock on one "
                         "laptop GPU under WDDM and are not a controlled benchmark; "
                         "they are comparable to each other because both runs used "
                         "the same machine and configuration.")}

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}")

    if args.markdown:
        os.makedirs(os.path.dirname(args.markdown) or ".", exist_ok=True)
        L = ["# Computational cost — measured values only", "",
             "| Metric | " + " | ".join(labels) + " |",
             "|---|" + "---:|" * len(labels)]
        for name, get in rows:
            L.append(f"| {name} | " + " | ".join(get(r) for r in records) + " |")
        L += ["", f"**GFLOPs:** {'measured' if have_flops else NA} — {flop_reason}. "
                  "Never estimated; an estimated FLOP count would be "
                  "indistinguishable from a measured one in this table.", ""]
        for r in records:
            for n in r["notes"]:
                L.append(f"* **{r['label']}**: {n}")
        L += ["", f"*{report['caveat']}*", ""]
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write("\n".join(L))
        print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
