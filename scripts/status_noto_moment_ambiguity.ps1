$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutputDir = Join-Path $Repo "data_work\noto\moment_constrained_active_fine_grid"
$StatusPath = Join-Path $OutputDir "run_status.json"
$DesignPath = Join-Path $OutputDir "run_design.json"
$PidPath = Join-Path $OutputDir "detached_run.pid"

if (-not (Test-Path -LiteralPath $StatusPath)) {
    Write-Output "No Noto moment-constrained status file found at $StatusPath"
    exit 0
}

$Status = Get-Content -Raw -LiteralPath $StatusPath | ConvertFrom-Json
$Design = if (Test-Path -LiteralPath $DesignPath) { Get-Content -Raw -LiteralPath $DesignPath | ConvertFrom-Json } else { $null }
$DetachedPid = if (Test-Path -LiteralPath $PidPath) { [int](Get-Content -Raw -LiteralPath $PidPath) } else { $null }
$WorkerPid = if ($Status.pid) { [int]$Status.pid } else { $null }
$DetachedRunning = if ($DetachedPid) { [bool](Get-Process -Id $DetachedPid -ErrorAction SilentlyContinue) } else { $false }
$WorkerRunning = if ($WorkerPid) { [bool](Get-Process -Id $WorkerPid -ErrorAction SilentlyContinue) } else { $false }
$Completed = if ($Status.completed_checkpoints) { [int]$Status.completed_checkpoints } else { 0 }
$Expected = if ($Design) { [int]$Design.expected_exact_evaluations } elseif ($Status.expected_evaluations) { [int]$Status.expected_evaluations } else { 0 }
$Percent = if ($Expected -gt 0) { 100.0 * $Completed / $Expected } else { 0.0 }

Write-Output "Status: $($Status.status)"
Write-Output "Block: $($Status.block)"
Write-Output "Message: $($Status.message)"
Write-Output ("Progress: {0}/{1} ({2:N1}%)" -f $Completed, $Expected, $Percent)
Write-Output "Detached PID: $DetachedPid running=$DetachedRunning"
Write-Output "Worker PID: $WorkerPid running=$WorkerRunning"
Write-Output "Output: $OutputDir"
Write-Output "Log: $($Status.log_path)"

$TablePath = Join-Path $OutputDir "tables\table_noto_moment_envelope_summary.csv"
if (Test-Path -LiteralPath $TablePath) {
    Write-Output ""
    Import-Csv -LiteralPath $TablePath |
        Select-Object rho,best_objective,baseline_capped_tv_objective,best_y_json,delta_rho_value,policy_changed_from_rho0,worst_failure_count_mean,active_moment_bounds |
        Format-Table -AutoSize |
        Out-String -Width 260 |
        Write-Output
}

if ($Status.log_path -and (Test-Path -LiteralPath $Status.log_path)) {
    Write-Output "Recent log lines:"
    Get-Content -LiteralPath $Status.log_path -Tail 20
}
