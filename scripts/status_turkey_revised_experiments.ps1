$ErrorActionPreference = "Stop"

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutputDir = Join-Path $Repo "data_work\turkey\revised_experiments"
$StatusPath = Join-Path $OutputDir "run_status.json"
$PidPath = Join-Path $OutputDir "detached_run.pid"

if (-not (Test-Path -LiteralPath $StatusPath)) {
    Write-Output "No revised-experiments status file found at $StatusPath"
    exit 0
}

$Status = Get-Content -Raw -LiteralPath $StatusPath | ConvertFrom-Json
$PidValue = $null
if (Test-Path -LiteralPath $PidPath) {
    $PidValue = [int](Get-Content -Raw -LiteralPath $PidPath)
}

$PythonPid = $null
if ($Status.pid) {
    $PythonPid = [int]$Status.pid
}

$DetachedRunning = $false
if ($PidValue) {
    $DetachedRunning = [bool](Get-Process -Id $PidValue -ErrorAction SilentlyContinue)
}

$PythonRunning = $false
if ($PythonPid) {
    $PythonRunning = [bool](Get-Process -Id $PythonPid -ErrorAction SilentlyContinue)
}

Write-Output "Status: $($Status.status)"
Write-Output "Block: $($Status.block)"
Write-Output "Message: $($Status.message)"
Write-Output "Updated: $($Status.updated_at_epoch)"
Write-Output "Detached PID: $PidValue running=$DetachedRunning"
Write-Output "Worker PID: $PythonPid running=$PythonRunning"
Write-Output "Output: $($Status.output_dir)"
Write-Output "Checkpoints: $($Status.checkpoint_dir)"
Write-Output "Log: $($Status.log_path)"

if ($Status.checkpoint_dir -and (Test-Path -LiteralPath $Status.checkpoint_dir)) {
    $Discrete = Get-ChildItem -LiteralPath $Status.checkpoint_dir -Filter "*discretized_m5*" -ErrorAction SilentlyContinue
    if ($Discrete.Count -gt 0) {
        Write-Output ""
        Write-Output "Discretized m=5 checkpoints:"
        $Discrete |
            Group-Object {
                if ($_.Name -match "rho([0-9]+\.[0-9]+)") { $Matches[1] } else { "unknown" }
            } |
            Sort-Object Name |
            ForEach-Object {
                Write-Output ("  rho={0}: {1} files" -f $_.Name, $_.Count)
            }
        $LatestDiscrete = $Discrete | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        Write-Output "  latest: $($LatestDiscrete.Name) at $($LatestDiscrete.LastWriteTime)"
    }

    $BudgetSensitivity = Get-ChildItem -LiteralPath $Status.checkpoint_dir -Filter "*budget_sensitivity_m5*" -ErrorAction SilentlyContinue
    if ($BudgetSensitivity.Count -gt 0) {
        Write-Output ""
        Write-Output "Budget-sensitivity m=5 checkpoints:"
        $BudgetSensitivity |
            Group-Object {
                $multiplier = "unknown"
                $rho = "unknown"
                if ($_.Name -match "mult([0-9]+\.[0-9]+)") { $multiplier = $Matches[1] }
                if ($_.Name -match "rho([0-9]+\.[0-9]+)") { $rho = $Matches[1] }
                "mult=$multiplier rho=$rho"
            } |
            Sort-Object Name |
            ForEach-Object {
                Write-Output ("  {0}: {1} files" -f $_.Name, $_.Count)
            }
        $LatestBudgetSensitivity = $BudgetSensitivity | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        Write-Output "  latest: $($LatestBudgetSensitivity.Name) at $($LatestBudgetSensitivity.LastWriteTime)"
    }
}

if ($Status.log_path -and (Test-Path -LiteralPath $Status.log_path)) {
    Write-Output ""
    Write-Output "Recent log lines:"
    Get-Content -LiteralPath $Status.log_path -Tail 25
}
