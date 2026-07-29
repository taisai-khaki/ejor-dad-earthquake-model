param(
    [ValidateSet(2, 3)]
    [int]$NumLinks = 3
)

$ErrorActionPreference = "Stop"

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutputDir = Join-Path $Repo "data_work\noto\support_preserving_m$NumLinks"
$StatusPath = Join-Path $OutputDir "run_status.json"
$PidPath = Join-Path $OutputDir "detached_run.pid"

if (-not (Test-Path -LiteralPath $StatusPath)) {
    Write-Output "No support-preserving Noto m=$NumLinks status file found at $StatusPath"
    exit 0
}

$Status = Get-Content -Raw -LiteralPath $StatusPath | ConvertFrom-Json
$DetachedPid = if (Test-Path -LiteralPath $PidPath) { [int](Get-Content -Raw -LiteralPath $PidPath) } else { $null }
$WorkerPid = if ($Status.pid) { [int]$Status.pid } else { $null }
$DetachedRunning = if ($DetachedPid) { [bool](Get-Process -Id $DetachedPid -ErrorAction SilentlyContinue) } else { $false }
$WorkerRunning = if ($WorkerPid) { [bool](Get-Process -Id $WorkerPid -ErrorAction SilentlyContinue) } else { $false }

Write-Output "Status: $($Status.status)"
Write-Output "Block: $($Status.block)"
Write-Output "Message: $($Status.message)"
Write-Output "Detached PID: $DetachedPid running=$DetachedRunning"
Write-Output "Worker PID: $WorkerPid running=$WorkerRunning"
Write-Output "Output: $OutputDir"
Write-Output "Log: $($Status.log_path)"

$TablePath = Join-Path $OutputDir "tables\table_noto_support_m$NumLinks`_certification.csv"
if (Test-Path -LiteralPath $TablePath) {
    Write-Output ""
    Write-Output "Completed rows:"
    Import-Csv -LiteralPath $TablePath |
        Select-Object rho,objective,lower_bound,absolute_gap,relative_gap_percent,selected_y_json,nodes_processed,converged,termination_reason |
        Format-Table -AutoSize |
        Out-String -Width 240 |
        Write-Output
}

if ($Status.log_path -and (Test-Path -LiteralPath $Status.log_path)) {
    Write-Output "Recent log lines:"
    Get-Content -LiteralPath $Status.log_path -Tail 25
}
