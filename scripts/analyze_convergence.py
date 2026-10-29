"""
Convergence analysis for a training run (Task 4 / Task 6).

Classifies the validation-PSNR trajectory into exactly one of:

    A  plateaued                  -> baseline ready for ablations
    B  still increasing           -> extend training
    C  peaked then declining      -> investigate overfitting / checkpoint choice
    D  unstable or invalid        -> stop and diagnose

The thresholds are stated explicitly below and applied mechanically, so the
verdict does not depend on eyeballing a curve.

    python scripts/analyze_convergence.py --experiment baseline_restormer
"""

import argparse
import csv
import json
import os
import sys

# Decision thresholds, in dB of validation PSNR.
# Rationale: on GoPro a <0.05 dB gain over the last ~10 epochs is well inside
# epoch-to-epoch noise (the 10-epoch run showed ~0.2 dB swings between adjacent
# epochs), whereas >0.15 dB of sustained gain is a real trend worth more compute.
PLATEAU_DB = 0.05
STRONG_DB = 0.15
DECLINE_DB = 0.10          # drop from peak that counts as genuine degradation


def to_float(v):
    try:
        f = float(v)
        return f if f == f else None      # reject NaN
    except (TypeError, ValueError):
        return None


def analyze(points, window=10, train_losses=None):
    """
    Classify a validation-PSNR trajectory.

    Args:
        points: iterable of (epoch, psnr). May be unsorted and may repeat an
            epoch (appended logs from resumed runs do both).
        window: trailing epochs used to judge the trend.
        train_losses: optional train-loss series, used only for the D verdict.

    Returns a dict including verdict_code in {A,B,C,D}.
    """
    raw = [(int(e), p) for e, p in points if p is not None and p == p]
    if len(raw) < 3:
        raise ValueError(f"need >= 3 validated epochs, got {len(raw)}")

    # Keep the LAST row per epoch, then sort. Trusting file order would make
    # rows[-1] the wrong epoch for a resumed or smoke-test-contaminated log.
    by_epoch = {}
    for e, psnr_v in raw:
        by_epoch[e] = psnr_v
    pts = sorted(by_epoch.items())

    epochs = [e for e, _ in pts]
    psnr = [p for _, p in pts]
    last_epoch, final = epochs[-1], psnr[-1]
    peak = max(psnr)
    peak_epoch = epochs[psnr.index(peak)]

    w = min(window, len(pts) - 1)
    gain_window = final - psnr[-(w + 1)]

    third = max(len(psnr) // 3, 1)
    recent_mean = sum(psnr[-third:]) / third
    prior_mean = (sum(psnr[-2 * third:-third]) / third
                  if len(psnr) >= 2 * third else None)
    mean_gain = (recent_mean - prior_mean) if prior_mean is not None else None
    drop_from_peak = peak - final

    losses = [x for x in (train_losses or []) if x is not None]
    invalid = (train_losses is not None and not losses) or any(x != x for x in losses)

    # Order matters. Decline is checked before plateau because a large NEGATIVE
    # gain also satisfies "gain <= plateau threshold" -- testing that naively
    # reports a collapsing run as plateaued and would greenlight ablations.
    if invalid:
        verdict, code = "Training is unstable or metrics are invalid.", "D"
    elif gain_window < -PLATEAU_DB or (drop_from_peak > DECLINE_DB
                                       and peak_epoch < last_epoch - 2):
        verdict, code = "Validation PSNR peaked and is now declining.", "C"
    elif gain_window >= STRONG_DB or (mean_gain is not None and mean_gain >= STRONG_DB):
        verdict, code = "Validation PSNR is still increasing substantially.", "B"
    elif abs(gain_window) <= PLATEAU_DB and (mean_gain is None
                                             or abs(mean_gain) <= PLATEAU_DB):
        verdict, code = "Validation PSNR has plateaued.", "A"
    else:
        verdict, code = ("Validation PSNR is increasing slowly (between plateau and "
                         "strong growth)."), "B"

    return {
        "points": pts, "epochs_validated": len(pts), "final_epoch": last_epoch,
        "final_psnr": final, "best_psnr": peak, "best_epoch": peak_epoch,
        "drop_from_peak_db": drop_from_peak, "window": w,
        "gain_over_window_db": gain_window, "gain_per_epoch_db": gain_window / w,
        "mean_gain_db": mean_gain, "recent_mean": recent_mean, "prior_mean": prior_mean,
        "duplicates_or_unsorted": len(raw) != len(pts) or any(
            raw[i][0] > raw[i + 1][0] for i in range(len(raw) - 1)),
        "raw_rows": len(raw),
        "thresholds": {"plateau_db": PLATEAU_DB, "strong_db": STRONG_DB,
                       "decline_db": DECLINE_DB},
        "verdict_code": code, "verdict": verdict,
        "baseline_ready_for_ablations": code == "A",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="baseline_restormer")
    ap.add_argument("--log", default=None)
    ap.add_argument("--window", type=int, default=10,
                    help="epochs at the end of the run used to judge the trend")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    log = args.log or os.path.join("logs", args.experiment, "train_log.csv")
    if not os.path.exists(log):
        print(f"No log at {log}")
        return 1

    with open(log, newline="") as f:
        rows = list(csv.DictReader(f))

    raw = [(int(r["epoch"]), to_float(r.get("val_psnr"))) for r in rows]
    raw = [(e, p) for e, p in raw if p is not None]
    if len(raw) < 3:
        print(f"Only {len(raw)} validated epoch(s) -- not enough to judge a trend.")
        return 1

    # A CSV is appended to, so a resumed run (or a smoke test sharing the log)
    # can leave rows out of order or repeat an epoch. Taking rows[-1] as "final"
    # would then read the wrong epoch. Sort by epoch and keep the LAST row for
    # each, which is the most recent measurement of that epoch.
    by_epoch = {}
    for e, psnr_v in raw:
        by_epoch[e] = psnr_v
    pts = sorted(by_epoch.items())

    n_dupes = len(raw) - len(pts)
    out_of_order = any(raw[i][0] > raw[i + 1][0] for i in range(len(raw) - 1))
    if n_dupes or out_of_order:
        print(f"NOTE: log contained {len(raw)} rows -> {len(pts)} unique epochs "
              f"({n_dupes} duplicate/superseded"
              + (", rows out of order" if out_of_order else "") + "). "
              "Using the last row per epoch, sorted.")

    epochs = [e for e, _ in pts]
    psnr = [p for _, p in pts]
    last_epoch = epochs[-1]
    final = psnr[-1]
    peak = max(psnr)
    peak_epoch = epochs[psnr.index(peak)]

    # Trend over the trailing window.
    w = min(args.window, len(pts) - 1)
    gain_window = final - psnr[-(w + 1)]
    per_epoch = gain_window / w

    # Robustness: compare mean of last third vs the third before it, which is
    # far less sensitive to a single noisy epoch than endpoint differencing.
    third = max(len(psnr) // 3, 1)
    recent_mean = sum(psnr[-third:]) / third
    prior_mean = sum(psnr[-2 * third:-third]) / third if len(psnr) >= 2 * third else None
    mean_gain = (recent_mean - prior_mean) if prior_mean is not None else None

    drop_from_peak = peak - final
    losses = [to_float(r.get("train_loss")) for r in rows]
    losses = [x for x in losses if x is not None]
    nonfinite = any(to_float(r.get("val_psnr")) is None for r in rows if r.get("val_psnr"))

    # ---- classification ----
    # Order matters. Decline is checked before plateau because a large NEGATIVE
    # gain also satisfies "gain <= plateau threshold" -- testing that naively
    # would report a collapsing run as plateaued and greenlight ablations.
    if nonfinite or not losses or any(x != x for x in losses):
        verdict, code = "Training is unstable or metrics are invalid.", "D"
    elif gain_window < -PLATEAU_DB or (drop_from_peak > DECLINE_DB
                                       and peak_epoch < last_epoch - 2):
        verdict, code = "Validation PSNR peaked and is now declining.", "C"
    elif gain_window >= STRONG_DB or (mean_gain is not None and mean_gain >= STRONG_DB):
        verdict, code = "Validation PSNR is still increasing substantially.", "B"
    elif abs(gain_window) <= PLATEAU_DB and (mean_gain is None
                                             or abs(mean_gain) <= PLATEAU_DB):
        verdict, code = "Validation PSNR has plateaued.", "A"
    else:
        verdict, code = ("Validation PSNR is increasing slowly (between plateau and "
                         "strong growth)."), "B"

    ready = code == "A"

    print("=" * 72)
    print(f"CONVERGENCE ANALYSIS -- {args.experiment}")
    print("=" * 72)
    print(f"epochs validated      : {len(pts)}  (0..{last_epoch})")
    print(f"final val PSNR        : {final:.4f} dB (epoch {last_epoch})")
    print(f"best  val PSNR        : {peak:.4f} dB (epoch {peak_epoch})")
    print(f"drop from peak        : {drop_from_peak:.4f} dB")
    print("-" * 72)

    print("trajectory:")
    for idx in sorted({len(pts) - 1 - w, len(pts) - 1 - w // 2, len(pts) - 1}):
        if 0 <= idx < len(pts):
            ee, pp = pts[idx]
            print(f"  epoch {ee:>3}: {pp:.4f} dB")

    print("-" * 72)
    print(f"gain over last {w} epochs : {gain_window:+.4f} dB  ({per_epoch:+.4f} dB/epoch)")
    if mean_gain is not None:
        print(f"mean of last {third} vs prior {third}: {recent_mean:.4f} vs {prior_mean:.4f} "
              f"({mean_gain:+.4f} dB)")
    print(f"thresholds            : plateau <= {PLATEAU_DB} dB, strong >= {STRONG_DB} dB, "
          f"decline > {DECLINE_DB} dB")
    print("=" * 72)
    print(f"VERDICT {code}: {verdict}")
    print(f"Baseline ready for ablations: {'YES' if ready else 'NO'}")
    print("=" * 72)

    out = {
        "experiment": args.experiment, "epochs_validated": len(pts),
        "final_epoch": last_epoch, "final_psnr": final,
        "best_psnr": peak, "best_epoch": peak_epoch,
        "drop_from_peak_db": drop_from_peak,
        "window": w, "gain_over_window_db": gain_window,
        "gain_per_epoch_db": per_epoch, "mean_gain_db": mean_gain,
        "thresholds": {"plateau_db": PLATEAU_DB, "strong_db": STRONG_DB,
                       "decline_db": DECLINE_DB},
        "verdict_code": code, "verdict": verdict,
        "baseline_ready_for_ablations": ready,
    }
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
