"""
Regression tests for the convergence verdict (scripts/analyze_convergence.py).

This tool decides "baseline ready for ablations: YES/NO". Two real bugs were
found by running it on a known trajectory, and both are pinned here:

  1. It trusted CSV row order. An appended log (resumed run, or a smoke test
     sharing the file) left a stale epoch-0 row last, so `rows[-1]` was read as
     the final epoch.
  2. The plateau test was `gain <= PLATEAU_DB`, which is also true for large
     NEGATIVE gains -- a run that collapsed by 4.9 dB was reported as
     "plateaued, ready for ablations: YES".
"""

import pytest

from scripts.analyze_convergence import DECLINE_DB, PLATEAU_DB, STRONG_DB, analyze


def flat(n=20, value=25.0):
    return [(i, value + (0.005 if i % 2 else -0.005)) for i in range(n)]


def rising(n=20, start=23.0, per_epoch=0.1):
    return [(i, start + i * per_epoch) for i in range(n)]


def declining(n=20, peak_at=8, top=25.0):
    return [(i, top - abs(i - peak_at) * 0.2) for i in range(n)]


# --------------------------------------------------------------------------
# verdict classification
# --------------------------------------------------------------------------

def test_plateau_gives_A_and_ready_yes():
    r = analyze(flat(), window=10)
    assert r["verdict_code"] == "A"
    assert r["baseline_ready_for_ablations"] is True


def test_strong_increase_gives_B_and_not_ready():
    r = analyze(rising(), window=10)
    assert r["verdict_code"] == "B"
    assert r["baseline_ready_for_ablations"] is False


def test_decline_gives_C_not_ready():
    r = analyze(declining(), window=10)
    assert r["verdict_code"] == "C"
    assert r["baseline_ready_for_ablations"] is False


def test_slow_increase_gives_B():
    """Between plateau and strong growth must not be mistaken for a plateau."""
    per_epoch = (PLATEAU_DB + STRONG_DB) / 2 / 10
    r = analyze(rising(n=20, per_epoch=per_epoch), window=10)
    assert r["verdict_code"] == "B"


def test_nan_train_loss_gives_D():
    r = analyze(flat(), window=10, train_losses=[0.2, float("nan"), 0.19])
    assert r["verdict_code"] == "D"
    assert r["baseline_ready_for_ablations"] is False


# --------------------------------------------------------------------------
# BUG 1 -- row order must not be trusted
# --------------------------------------------------------------------------

def test_stale_trailing_row_does_not_become_the_final_epoch():
    """
    The exact shape of the real contaminated log: epochs 0..9 followed by a
    stale epoch-0 row from a smoke test.
    """
    pts = [(i, 24.0 + i * 0.05) for i in range(10)] + [(0, 19.2)]
    r = analyze(pts, window=5)

    assert r["final_epoch"] == 9
    assert r["final_psnr"] == pytest.approx(24.45)
    assert r["duplicates_or_unsorted"] is True
    assert r["verdict_code"] != "C", "stale row must not fake a decline"


def test_shuffled_rows_give_same_result_as_sorted():
    ordered = rising(n=15)
    shuffled = [ordered[i] for i in (7, 0, 3, 14, 2, 9, 1, 12, 5, 4, 11, 6, 13, 8, 10)]
    assert analyze(shuffled, window=8)["verdict_code"] == analyze(ordered, window=8)["verdict_code"]
    assert analyze(shuffled, window=8)["final_epoch"] == 14


def test_duplicate_epoch_keeps_last_occurrence():
    """A resumed run re-logging an epoch: the newer measurement wins."""
    r = analyze([(0, 20.0), (1, 21.0), (2, 22.0), (2, 26.0)], window=2)
    assert r["final_psnr"] == pytest.approx(26.0)
    assert r["epochs_validated"] == 3


# --------------------------------------------------------------------------
# BUG 2 -- a collapse must never read as a plateau
# --------------------------------------------------------------------------

def test_large_negative_gain_is_not_a_plateau():
    """The regression: -4.89 dB satisfies `gain <= 0.05` but is a collapse."""
    pts = [(i, 24.5) for i in range(10)] + [(10, 19.6)]
    r = analyze(pts, window=5)

    assert r["gain_over_window_db"] < -1.0
    assert r["verdict_code"] == "C"
    assert r["baseline_ready_for_ablations"] is False


@pytest.mark.parametrize("drop", [0.06, 0.5, 2.0, 5.0])
def test_no_drop_size_is_ever_reported_ready(drop):
    pts = [(i, 25.0) for i in range(10)] + [(10, 25.0 - drop)]
    r = analyze(pts, window=5)
    assert r["baseline_ready_for_ablations"] is False


def test_tiny_negative_noise_still_counts_as_plateau():
    """Sub-threshold negative wobble is noise, not decline."""
    pts = [(i, 25.0) for i in range(10)] + [(10, 25.0 - PLATEAU_DB / 2)]
    r = analyze(pts, window=5)
    assert r["verdict_code"] == "A"


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------

def test_too_few_epochs_raises():
    with pytest.raises(ValueError):
        analyze([(0, 20.0), (1, 21.0)], window=5)


def test_nan_psnr_rows_are_dropped():
    r = analyze([(0, 20.0), (1, float("nan")), (2, 22.0), (3, 23.0)], window=2)
    assert r["epochs_validated"] == 3


def test_thresholds_are_ordered_sensibly():
    assert 0 < PLATEAU_DB < DECLINE_DB <= STRONG_DB


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
