param(
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $Repo "data_work\noto\practical_resilience"
}
$StatusPath = Join-Path $OutputDir "run_status.json"
$PidPath = Join-Path $OutputDir "detached_run.pid"

if (-not (Test-Path -LiteralPath $StatusPath)) {
    Write-Output "No practical Noto status file found at $StatusPath"
    exit 0
}

$Status = Get-Content -Raw -LiteralPath $StatusPath | ConvertFrom-Json
$DetachedPid = if (Test-Path -LiteralPath $PidPath) { [int](Get-Content -Raw -LiteralPath $PidPath) } else { $null }
$WorkerPid = if ($Status.pid) { [int]$Status.pid } else { $null }
$DetachedRunning = if ($DetachedPid) { [bool](Get-Process -Id $DetachedPid -ErrorAction SilentlyContinue) } else { $false }
$WorkerRunning = if ($WorkerPid) { [bool](Get-Process -Id $WorkerPid -ErrorAction SilentlyContinue) } else { $false }
$Completed = if ($Status.evaluated) { [int]$Status.evaluated } else { 0 }
$Expected = if ($Status.total_grid -and $null -ne $Status.infeasible) {
    [int]$Status.total_grid - [int]$Status.infeasible
} else {
    0
}
$Percent = if ($Expected -gt 0) { 100.0 * $Completed / $Expected } else { 0.0 }

Write-Output "Status: $($Status.status)"
Write-Output "Block: $($Status.block)"
Write-Output "Message: $($Status.message)"
if ($Expected -gt 0) {
    Write-Output ("Active-radius progress: {0}/{1} ({2:N1}%)" -f $Completed, $Expected, $Percent)
}
Write-Output "Detached PID: $DetachedPid running=$DetachedRunning"
Write-Output "Worker PID: $WorkerPid running=$WorkerRunning"
Write-Output "Output: $OutputDir"
Write-Output "Checkpoint directory: $($Status.checkpoint_dir)"
Write-Output "Log: $($Status.log_path)"

$SummaryPath = Join-Path $OutputDir "tables\table_noto_practical_summary.csv"
if (Test-Path -LiteralPath $SummaryPath) {
    Write-Output ""
    Import-Csv -LiteralPath $SummaryPath |
        Select-Object rho,best_discretized_objective,best_y_json,delta_rho_value,y_l1_distance_from_rho0,road_value_over_no_retrofit,runtime_sec |
        Format-Table -AutoSize |
        Out-String -Width 260 |
        Write-Output
}

if ($Status.log_path -and (Test-Path -LiteralPath $Status.log_path)) {
    Write-Output "Recent progress log lines:"
    Get-Content -LiteralPath $Status.log_path -Tail 20
}