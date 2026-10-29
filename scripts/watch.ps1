<#
    watch.ps1 -- live epoch-by-epoch view of the running chain.

    READ-ONLY. It opens no process, signals nothing, and touches no
    checkpoint: it polls the files the trainer writes and prints them.
    Ctrl+C here stops the watcher and NOTHING else -- the training keeps
    going. Safe to start and stop as often as you like.

    This exists because train.py's stdout is block-buffered when the chain
    redirects it to a file, so a healthy run prints nothing for hours.

    USAGE
        powershell -ExecutionPolicy Bypass -File scripts\watch.ps1
        powershell -ExecutionPolicy Bypass -File scripts\watch.ps1 -PollSeconds 10
#>

param(
    [int]$PollSeconds = 15,
    [int]$TargetEpochs = 60
)

$ErrorActionPreference = "SilentlyContinue"
Set-Location (Split-Path -Parent $PSScriptRoot)

$STAGES = @("seecoder_diffae_lcr_flow", "full_srrt", "seecoder")

# Which stage is live right now? Ask the process table, and fall back to
# the last stage that has a log if training is between stages.
function Get-ActiveStage {
    $t = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
           Where-Object { $_.CommandLine -like "*train.py*" -and $_.CommandLine -notlike "*.venv*" })
    if ($t.Count -and $t[0].CommandLine -match '--ablation\s+(\S+)') { return $Matches[1] }
    foreach ($s in ($STAGES | Sort-Object { $STAGES.IndexOf($_) } -Descending)) {
        if (Test-Path "logs\$s\train_log.csv") { return $s }
    }
    return $STAGES[0]
}

function Get-Rows($stage) {
    $p = "logs\$stage\train_log.csv"
    if (-not (Test-Path $p)) { return @() }
    return @(Import-Csv $p | Where-Object { $_.epoch -match '^\d+$' })
}

# Position WITHIN the current epoch, scraped from the trainer's stdout log
# ("  epoch 6 iter 720/1051 loss=0.1394 (2.34 it/s)"). Only meaningful when
# the chain set PYTHONUNBUFFERED=1; a buffered log is stale by ~8 KB, so
# anything older than a couple of minutes is ignored rather than shown.
function Get-LiveIter($stage) {
    $f = Get-ChildItem "logs\${stage}_chain_try*_out.log" -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime | Select-Object -Last 1
    if (-not $f) { return $null }
    if (((Get-Date) - $f.LastWriteTime).TotalMinutes -gt 2) { return $null }
    $tail = Get-Content $f.FullName -Tail 40 -ErrorAction SilentlyContinue
    if (-not $tail) { return $null }
    for ($i = $tail.Count - 1; $i -ge 0; $i--) {
        if ($tail[$i] -match 'epoch\s+(\d+)\s+iter\s+(\d+)/(\d+)\s+loss=([\d.]+)\s+\(([\d.]+)\s*it/s\)') {
            return @{ epoch = [int]$Matches[1]; iter = [int]$Matches[2]; total = [int]$Matches[3]
                      loss  = [double]$Matches[4]; rate = [double]$Matches[5] }
        }
    }
    return $null
}

Write-Host ""
Write-Host "  watching SRRT chain -- Ctrl+C stops THIS VIEW ONLY, not the training" -ForegroundColor Cyan
Write-Host "  poll every ${PollSeconds}s" -ForegroundColor DarkGray
Write-Host ""

$seen = @{}
$lastStage = ""

# Print history already on disk, so the view starts with full context
# rather than waiting up to 7 minutes for the next epoch to land.
$stage = Get-ActiveStage
foreach ($r in (Get-Rows $stage)) {
    $key = "$stage/$($r.epoch)"
    $seen[$key] = $true
}
$hist = Get-Rows $stage
if ($hist.Count) {
    Write-Host "  [$stage] already completed:" -ForegroundColor Yellow
    foreach ($r in ($hist | Select-Object -Last 5)) {
        Write-Host ("    ep {0,2}/{1}   psnr {2,7:N3}   ssim {3,6:N4}   {4,5:N1} min" -f `
            [int]$r.epoch, $TargetEpochs, [double]$r.val_psnr, [double]$r.val_ssim, ([double]$r.epoch_time_s / 60))
    }
    Write-Host ""
}
$lastStage = $stage

while ($true) {
    $stage = Get-ActiveStage

    if ($stage -ne $lastStage) {
        Write-Host ""
        Write-Host "  >>> stage changed: $lastStage -> $stage" -ForegroundColor Cyan
        Write-Host ""
        $lastStage = $stage
    }

    $rows = Get-Rows $stage
    foreach ($r in $rows) {
        $key = "$stage/$($r.epoch)"
        if (-not $seen.ContainsKey($key)) {
            $seen[$key] = $true
            $ep = [int]$r.epoch
            $left = [Math]::Max(0, $TargetEpochs - $ep - 1)
            $eta = (Get-Date).AddMinutes($left * ([double]$r.epoch_time_s / 60))
            Write-Host ("  {0}  [{1}] ep {2,2}/{3}  psnr {4,7:N3}  ssim {5,6:N4}  lpips {6,6:N4}  {7,4:N1} min  | {8} left, eta {9}" -f `
                (Get-Date -Format "HH:mm:ss"), $stage, $ep, $TargetEpochs,
                [double]$r.val_psnr, [double]$r.val_ssim, [double]$r.val_lpips,
                ([double]$r.epoch_time_s / 60), $left, $eta.ToString("ddd HH:mm")) -ForegroundColor Green
        }
    }

    # Heartbeat between epoch boundaries, rewritten in place so the line
    # visibly moves even though an epoch takes minutes to land.
    $done = 0
    if ($rows.Count) { $done = ($rows | ForEach-Object { [int]$_.epoch } | Measure-Object -Maximum).Maximum + 1 }
    $logFile = Get-Item "logs\$stage\train_log.csv"
    $ago = 0.0
    if ($logFile) { $ago = ((Get-Date) - $logFile.LastWriteTime).TotalMinutes }

    $gpu = (nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader)
    if (-not $gpu) { $gpu = "gpu n/a" }

    $alive = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
               Where-Object { $_.CommandLine -like "*train.py*" -or $_.CommandLine -like "*evaluate.py*" })
    $state = "training"
    if ($alive.Count -eq 0) { $state = "NO TRAINER" }
    elseif ($alive[0].CommandLine -like "*evaluate.py*") { $state = "evaluating" }

    # A freshly (re)started stage has no new CSV row yet, so its last row
    # can be hours old while training is perfectly healthy. The stdout log
    # is the faster signal -- if it moved recently, we are not stalled.
    $outAge = 999.0
    $of = Get-ChildItem "logs\${stage}_chain_try*_out.log" -ErrorAction SilentlyContinue |
          Sort-Object LastWriteTime | Select-Object -Last 1
    if ($of) { $outAge = ((Get-Date) - $of.LastWriteTime).TotalMinutes }

    $warn = ""
    if ($state -eq "training" -and $ago -gt 20 -and $outAge -gt 5) { $warn = "  <-- STALLED?" }
    elseif ($state -eq "training" -and $ago -gt 20) { $warn = "  (epoch in progress)" }

    if ($state -eq "evaluating") {
        # No trainer running, so epoch timestamps are legitimately old.
        # Showing "last epoch N min ago" here reads as a stall; don't.
        $line = ("  {0}  evaluating (1111 images, ~13 min)  | gpu {1}    " -f `
                 (Get-Date -Format "HH:mm:ss"), $gpu)
    } else {
        $live = Get-LiveIter $stage
        if ($live) {
            $pct = 100.0 * $live.iter / $live.total
            $eta = ($live.total - $live.iter) / [Math]::Max($live.rate, 0.01) / 60.0
            $line = ("  {0}  ep {1}/{2}  iter {3,4}/{4}  {5,5:N1}%  loss {6:N4}  {7:N2} it/s  ~{8:N1} min to epoch end  | gpu {9}    " -f `
                     (Get-Date -Format "HH:mm:ss"), ($live.epoch + 1), $TargetEpochs,
                     $live.iter, $live.total, $pct, $live.loss, $live.rate, $eta, $gpu)
        } else {
            $line = ("  {0}  {1}  {2}/{3} done  | gpu {4} | last epoch {5:N1} min ago{6}    " -f `
                     (Get-Date -Format "HH:mm:ss"), $state, $done, $TargetEpochs, $gpu, $ago, $warn)
        }
    }
    Write-Host "`r$line" -NoNewline -ForegroundColor DarkGray

    Start-Sleep -Seconds $PollSeconds
}
