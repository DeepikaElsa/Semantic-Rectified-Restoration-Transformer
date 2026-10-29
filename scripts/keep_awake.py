"""
Suppress Windows idle standby for as long as a watched process is alive.

WHY THIS EXISTS (not part of any experiment)
--------------------------------------------
Long baseline/ablation runs on this laptop were repeatedly destroyed mid-epoch by
Modern Standby. Confirmed from the Windows event log:

    16:04:49  last training log line (epoch 22, iter 280)
    16:05:13  Kernel-Power 506 "The system is entering Modern Standby.
              Reason: Idle Timeout."

Detaching the trainer does not help: Modern Standby suspends the whole system, so
the process is frozen rather than killed and its CUDA context does not survive.
The same signature explains earlier 17,303 s and 5,515 s "epochs" in
logs/baseline_restormer/train_log.csv -- those are sleep durations, not compute.

This helper touches NOTHING in the experiment: no config, no checkpoint, no CSV,
no model code. It only asks Windows to treat the system as busy.

WHY SetThreadExecutionState AND NOT powercfg
--------------------------------------------
`powercfg /change standby-timeout-ac 0` would permanently rewrite the user's power
plan and silently persist after the run. SetThreadExecutionState is scoped to this
process: the request is dropped the moment the process exits, for any reason,
including being killed. Nothing to clean up and nothing left behind.

Flags used:
    ES_CONTINUOUS       0x80000000  keep the state until it is cleared
    ES_SYSTEM_REQUIRED  0x00000001  system is in use -> do not idle-sleep

ES_DISPLAY_REQUIRED is deliberately NOT set: the screen may still blank, which
saves power and does not affect compute.

LIMITATIONS (documented rather than papered over)
-------------------------------------------------
* Closing the lid still sleeps the machine if the lid action is "Sleep". This
  cannot be overridden without changing the user's settings, which is out of scope.
* A user-initiated sleep, hibernate, reboot or power loss still stops training.
* This does not make training resumable; train.py --resume already handles that.

    python scripts/keep_awake.py --pid 12345
    python scripts/keep_awake.py --pid 12345 --max-hours 12
"""

import argparse
import ctypes
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def pid_alive(pid: int) -> bool:
    """True while `pid` exists. psutil is already a project dependency."""
    import psutil
    try:
        proc = psutil.Process(pid)
        # A zombie still "exists" but is finished; treat it as gone.
        return proc.status() != psutil.STATUS_DEAD
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--pid", type=int, default=None,
                    help="exit once this process exits; omit to run until killed")
    ap.add_argument("--max-hours", type=float, default=24.0,
                    help="hard stop so a stray helper cannot keep the machine awake forever")
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    args = ap.parse_args()

    if not sys.platform.startswith("win"):
        print("keep_awake: not Windows, nothing to do.")
        return 0

    if args.pid is not None and not pid_alive(args.pid):
        print(f"keep_awake: pid {args.pid} is not running; refusing to start.")
        return 1

    set_state = ctypes.windll.kernel32.SetThreadExecutionState
    if set_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) == 0:
        print("keep_awake: SetThreadExecutionState failed; idle standby NOT suppressed.")
        return 1

    started = time.time()
    print(f"keep_awake: idle standby suppressed (watching pid={args.pid}, "
          f"max {args.max_hours} h). Lid-close sleep is NOT suppressed.")
    sys.stdout.flush()

    try:
        while True:
            time.sleep(args.poll_seconds)
            elapsed_h = (time.time() - started) / 3600.0
            if elapsed_h >= args.max_hours:
                print(f"keep_awake: reached --max-hours={args.max_hours}, releasing.")
                break
            if args.pid is not None and not pid_alive(args.pid):
                print(f"keep_awake: pid {args.pid} exited after {elapsed_h:.2f} h, releasing.")
                break
    except KeyboardInterrupt:
        print("keep_awake: interrupted, releasing.")
    finally:
        # Drop the request explicitly. Windows would drop it at process exit
        # anyway, but being explicit makes the lifetime obvious.
        set_state(ES_CONTINUOUS)
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
