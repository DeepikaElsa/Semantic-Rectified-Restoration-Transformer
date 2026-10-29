<#
    status.ps1 -- is the ablation chain alive, and where is it?

    Exists because train.py's stdout is block-buffered when redirected to a
    file, so logs\*_chain_*_out.log stays empty for long stretches and a
    perfectly healthy run looks dead. None of the checks below rely on
    stdout: they read the process table, the GPU, and the files the trainer
    actually writes once per completed epoch.

    USAGE
        powershell -ExecutionPolicy Bypass -File scripts\status.ps1
#>

$ErrorActionPreference = "SilentlyContinue"
Set-Location (Split-Path -Parent $PSScriptRoot)

$STAGES = @("seecoder_diffae_lcr_flow", "full_srrt", "seecoder")
$TARGET = 60

function Hr($t) { Write-Host ""; Write-Host "-- $t " -NoNewline; Write-Host ("-" * [Math]::Max(0, 62 - $t.Length)) }

Write-Host ""
Write-Host "  SRRT chain status  --  $(Get-Date -Format 'ddd dd MMM HH:mm:ss')"

# ---------------------------------------------------------------- processes
Hr "processes"
$chain = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -like "*run_remaining*" })
$train = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*train.py*" })
$evalp = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*evaluate.py*" })

if ($chain.Count) { Write-Host "  chain    RUNNING  pid $($chain[0].ProcessId)" -ForegroundColor Green }
else              { Write-Host "  chain    NOT RUNNING" -ForegroundColor Red }

if ($train.Count) {
    # .venv\Scripts\python.exe is a redirector that re-execs the base
    # interpreter; it burns no CPU, so measuring it always reads "stalled".
    # -like takes WILDCARDS, not regex -- "*.venv*" is the correct form.
    $real = $train | Where-Object { $_.CommandLine -notlike "*.venv*" } | Select-Object -First 1
    if (-not $real) { $real = $train[0] }
    $abl = if ($real.CommandLine -match '--ablation\s+(\S+)') { $Matches[1] } else { "?" }
    Write-Host "  training RUNNING  pid $($real.ProcessId)  stage: $abl" -ForegroundColor Green
} elseif ($evalp.Count) {
    Write-Host "  evaluating  pid $($evalp[0].ProcessId)" -ForegroundColor Green
} else {
    Write-Host "  training NOT RUNNING" -ForegroundColor Yellow
}

# CPU time must advance, or the process is wedged rather than working.
if ($train.Count) {
    $pid2 = ($train | Where-Object { $_.CommandLine -notlike "*.venv*" } | Select-Object -First 1).ProcessId
    $pr = Get-Process -Id $pid2
    if ($pr) {
        $c1 = $pr.CPU; Start-Sleep -Seconds 5; $pr.Refresh()
        $d = $pr.CPU - $c1
        $verdict = if ($d -gt 1) { "COMPUTING" } else { "STALLED?" }
        $col = if ($d -gt 1) { "Green" } else { "Red" }
        Write-Host ("  cpu      +{0:N1}s in 5s wall  -> {1}" -f $d, $verdict) -ForegroundColor $col
    }
}

# ---------------------------------------------------------------- gpu
Hr "gpu"
$g = (nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader)
if ($g) { Write-Host "  $g" } else { Write-Host "  nvidia-smi unavailable" }

# ---------------------------------------------------------------- progress
Hr "progress"
$totalRemaining = 0.0
foreach ($s in $STAGES) {
    $log = "logs\$s\train_log.csv"
    if (-not (Test-Path $log)) {
        Write-Host ("  {0,-26} not started" -f $s)
        $totalRemaining += $TARGET * 7.0
        continue
    }
    $rows = @(Import-Csv $log | Where-Object { $_.epoch -match '^\d+$' })
    if ($rows.Count -eq 0) { Write-Host ("  {0,-26} starting up" -f $s); continue }

    $done = ($rows | ForEach-Object { [int]$_.epoch } | Measure-Object -Maximum).Maximum + 1
    $psnr = [double]($rows[-1].val_psnr)

    # Median over recent epochs: immune to a frozen epoch inflating the mean.
    $recent = @($rows | Select-Object -Last 10 | ForEach-Object { [double]$_.epoch_time_s } | Sort-Object)
    $medMin = if ($recent.Count) { $recent[[int]($recent.Count/2)] / 60.0 } else { 7.0 }

    $left = [Math]::Max(0, $TARGET - $done)
    $totalRemaining += $left * $medMin

    $bar = "#" * [int](24 * $done / $TARGET)
    $tag = if ($left -eq 0) { "DONE" } else { "{0:N1}h left" -f ($left * $medMin / 60) }
    Write-Host ("  {0,-26} [{1,-24}] {2,2}/{3}  psnr {4,6:N3}  {5}" -f $s, $bar, $done, $TARGET, $psnr, $tag)
}

Write-Host ""
Write-Host ("  ALL STAGES DONE ~ {0}   ({1:N1} h remaining)" -f `
            (Get-Date).AddHours($totalRemaining/60 + 0.2).ToString("ddd dd MMM HH:mm"), ($totalRemaining/60))

# ---------------------------------------------------------------- freshness
Hr "last written"
# Follow the ACTIVE stage, not a hardcoded one: a finished stage's log is
# legitimately hours old and must not be read as a stall.
$active = $STAGES[0]
$tp = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*train.py*" -and $_.CommandLine -notlike "*.venv*" })
if ($tp.Count -and $tp[0].CommandLine -match '--ablation\s+(\S+)') { $active = $Matches[1] }
Write-Host "  (active stage: $active)" -ForegroundColor DarkGray
foreach ($f in @("logs\CHAIN_STATUS.log","logs\$active\train_log.csv","checkpoints\$active\latest.pth")) {
    $i = Get-Item $f
    if ($i) {
        $ago = ((Get-Date) - $i.LastWriteTime).TotalMinutes
        $col = if ($ago -gt 20) { "Yellow" } else { "Gray" }
        Write-Host ("  {0,-52} {1,5:N1} min ago" -f $i.Name, $ago) -ForegroundColor $col
    }
}

Hr "chain log (last 6)"
Get-Content "logs\CHAIN_STATUS.log" -Tail 6 | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
Write-Host ""
