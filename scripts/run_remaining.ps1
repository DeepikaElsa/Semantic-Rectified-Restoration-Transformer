<#
    run_remaining.ps1 -- finish the SRRT ablation ladder unattended.

    Waits for any in-flight train.py to finish, then runs the remaining
    stages to completion, evaluating each one:

        E  seecoder_diffae_lcr_flow   (resumed if the in-flight run died)
        F  full_srrt
        B  seecoder

    Every stage runs 60 epochs with --epochs 60, which also retunes the
    cosine T_max so the schedule fully anneals. That matching of schedule
    across arms is the whole point of the reruns: without it the arms are
    not comparable and the rows are inadmissible.

    SAFETY
    ------
    * Pre-fix result directories are archived, never overwritten. The
      broken E and F numbers are cited in the paper's failure analysis.
    * A stage that exits early is retried with --resume, up to -MaxRetries.
    * Completion is judged from train_log.csv, not from exit codes: a
      process can exit 0 having done nothing useful.
    * A stage already at its target epoch count is skipped, so the script
      is safe to re-run after an interruption.
    * keep_awake.py is held for the whole chain, watching THIS process.

    USAGE
        powershell -ExecutionPolicy Bypass -File scripts\run_remaining.ps1
#>

param(
    [int]$MaxRetries    = 3,
    [int]$TargetEpochs  = 60,
    [double]$MaxHours   = 30.0,
    # Skip keep_awake.py. Correct ONLY when idle standby is already off at
    # the OS level -- check with:
    #     powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE
    # "Current AC Power Setting Index: 0x00000000" means Never, and the
    # helper is redundant. Any other value and the machine will sleep
    # mid-epoch; see the incident recorded in scripts/keep_awake.py.
    [switch]$NoKeepAwake
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Py     = ".venv\Scripts\python.exe"
$Status = "logs\CHAIN_STATUS.log"

function Write-Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $Status -Value $line -Encoding utf8
    Write-Host $line
}

# Highest epoch index recorded in a run's train_log.csv, or -1 if none.
# This is the source of truth for "did the stage finish", because the log
# is appended once per completed epoch.
function Get-LastEpoch($name) {
    $p = "logs\$name\train_log.csv"
    if (-not (Test-Path $p)) { return -1 }
    try {
        $rows = @(Import-Csv $p | Where-Object { $_.epoch -match '^\d+$' })
    } catch { return -1 }
    if ($rows.Count -eq 0) { return -1 }
    return ($rows | ForEach-Object { [int]$_.epoch } | Measure-Object -Maximum).Maximum
}

function Test-StageComplete($name) {
    return (Get-LastEpoch $name) -ge ($TargetEpochs - 1)
}

# Move a directory aside with a timestamp, once, so nothing is clobbered.
function Move-Aside($path, $tag) {
    if (-not (Test-Path $path)) { return }
    $dest = "${path}_${tag}"
    # Archive ONCE. On a chain restart $path holds a freshly computed
    # result, not the pre-fix one -- moving it again would file a good
    # result under a "BROKEN" name and force a needless re-evaluation.
    if (Test-Path $dest) {
        Write-Log "archive $dest already exists -- leaving $path alone"
        return
    }
    Move-Item $path $dest
    Write-Log "archived $path -> $dest"
}

function Invoke-Stage($name) {
    if (Test-StageComplete $name) {
        Write-Log "$name : already complete at epoch $(Get-LastEpoch $name) -- skipping"
        return $true
    }

    for ($try = 1; $try -le $MaxRetries; $try++) {
        $last   = Get-LastEpoch $name
        $argStr = "train.py --ablation $name --epochs $TargetEpochs"

        # Resume only if there is something to resume from. A fresh stage
        # has no latest.pth and must start clean.
        $ckpt = "checkpoints\$name\latest.pth"
        if (Test-Path $ckpt) { $argStr += " --resume $ckpt" }

        Write-Log "$name : attempt $try/$MaxRetries, last completed epoch $last"
        Write-Log "$name : $Py $argStr"

        $proc = Start-Process $Py -ArgumentList $argStr -NoNewWindow -PassThru `
                    -RedirectStandardOutput "logs\${name}_chain_try${try}_out.log" `
                    -RedirectStandardError  "logs\${name}_chain_try${try}_err.log"
        $proc.WaitForExit()

        $last = Get-LastEpoch $name
        if (Test-StageComplete $name) {
            Write-Log "$name : COMPLETE at epoch $last (exit $($proc.ExitCode))"
            return $true
        }
        Write-Log "$name : exited early at epoch $last (exit $($proc.ExitCode)) -- will retry"
    }

    Write-Log "$name : FAILED -- gave up after $MaxRetries attempts"
    return $false
}

function Invoke-Eval($name) {
    $ckpt = "checkpoints\$name\best_psnr.pth"
    if (-not (Test-Path $ckpt)) {
        Write-Log "$name : no best_psnr.pth -- skipping eval"
        return
    }
    Write-Log "$name : evaluating $ckpt"
    $proc = Start-Process $Py -ArgumentList "evaluate.py --checkpoint $ckpt" -NoNewWindow -PassThru `
                -RedirectStandardOutput "logs\${name}_eval_out.log" `
                -RedirectStandardError  "logs\${name}_eval_err.log"
    $proc.WaitForExit()

    $summary = "results\$name\summary.json"
    if (Test-Path $summary) {
        $s = Get-Content $summary -Raw | ConvertFrom-Json
        Write-Log ("$name : EVAL psnr={0:N3} ssim={1:N4} lpips={2:N4} (epoch {3}, {4} images)" -f `
                   $s.metrics.psnr.mean, $s.metrics.ssim.mean, $s.metrics.lpips.mean, `
                   $s.epoch, $s.num_images)
    } else {
        Write-Log "$name : eval finished (exit $($proc.ExitCode)) but no summary.json found"
    }
}

# Block until no train.py is running. Called once, before this script
# launches anything of its own, so it can only ever see a foreign trainer.
function Wait-ForInFlightTrainer {
    $announced = $false
    while ($true) {
        $running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
                     Where-Object { $_.CommandLine -and $_.CommandLine -like "*train.py*" })
        if ($running.Count -eq 0) { break }
        if (-not $announced) {
            Write-Log "waiting for in-flight training (pids: $($running.ProcessId -join ', '))"
            $announced = $true
        }
        Start-Sleep -Seconds 60
    }
    if ($announced) { Write-Log "in-flight training finished" }
}

# ---------------------------------------------------------------- main

# REFUSE TO RUN TWICE. Two chains race on the same checkpoint dirs and
# append to the same train_log.csv, which silently corrupts a run: two
# trainers writing latest.pth and interleaving epoch rows. Launching the
# chain again while one is up is an easy mistake -- this makes it safe.
$others = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -like "*run_remaining*" -and $_.ProcessId -ne $PID })
if ($others.Count -gt 0) {
    $ids = ($others.ProcessId -join ', ')
    Write-Log "REFUSING TO START: chain already running (pid $ids). Nothing was changed."
    Write-Host ""
    Write-Host "  A chain is already running (pid $ids)." -ForegroundColor Yellow
    Write-Host "  Watch it:  powershell -ExecutionPolicy Bypass -File scripts\status.ps1" -ForegroundColor Yellow
    Write-Host "  Stop it :  Stop-Process -Id $ids -Force" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

Write-Log "=========================================================="
Write-Log "chain starting (pid $PID) -- target $TargetEpochs epochs/stage"

# Python block-buffers stdout when it is redirected to a file, so the
# per-iteration progress lines ("epoch 2 iter 720/1051 ...") arrive in
# 8 KB gulps -- several epochs late. Unbuffered, they land immediately
# and `Get-Content -Wait` on the out-log becomes a true live feed.
# Start-Process inherits this, so every stage and eval below is covered.
$env:PYTHONUNBUFFERED = "1"
Write-Log "PYTHONUNBUFFERED=1 (live stdout for tailing)"

# Hold off idle standby for the whole chain. Watching $PID means the
# request is released the moment this script exits, for any reason.
$ka = $null
if ($NoKeepAwake) {
    $idle = (powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | Select-String "Current AC")
    Write-Log "keep_awake SKIPPED (-NoKeepAwake). OS idle-sleep on AC: $(($idle -split ':')[-1].Trim())"
} else {
    $ka = Start-Process $Py -ArgumentList "scripts\keep_awake.py --pid $PID --max-hours $MaxHours" `
              -NoNewWindow -PassThru `
              -RedirectStandardOutput "logs\chain_keepawake.log" `
              -RedirectStandardError  "logs\chain_keepawake_err.log"
    Write-Log "keep_awake started (pid $($ka.Id)), max $MaxHours h"
}

try {
    Wait-ForInFlightTrainer

    # The pre-fix E and F evaluations are cited in the paper's failure
    # analysis. evaluate.py would overwrite results\<name>\, so park them.
    Move-Aside "results\seecoder_diffae_lcr_flow" "PREFIX_BROKEN"
    Move-Aside "results\full_srrt"                "PREFIX_BROKEN"

    $stages = @("seecoder_diffae_lcr_flow", "full_srrt", "seecoder")

    foreach ($stage in $stages) {
        if (Invoke-Stage $stage) {
            Invoke-Eval $stage
        } else {
            Write-Log "$stage : training failed -- skipping eval, continuing to next stage"
        }
    }

    Write-Log "---------------- FINAL SUMMARY ----------------"
    foreach ($name in @("restormer_baseline","seecoder","seecoder_diffae","seecoder_diffae_lcr","seecoder_diffae_lcr_flow","full_srrt")) {
        $summary = "results\$name\summary.json"
        if (Test-Path $summary) {
            $s = Get-Content $summary -Raw | ConvertFrom-Json
            Write-Log ("{0,-26} ep {1,3}  psnr {2,7:N3}  ssim {3,7:N4}  lpips {4,7:N4}" -f `
                       $name, $s.epoch, $s.metrics.psnr.mean, $s.metrics.ssim.mean, $s.metrics.lpips.mean)
        } else {
            Write-Log ("{0,-26} (no results)" -f $name)
        }
    }
    Write-Log "chain finished"
}
finally {
    if ($ka -and -not $ka.HasExited) {
        Stop-Process -Id $ka.Id -Force -ErrorAction SilentlyContinue
        Write-Log "keep_awake released"
    }
}
