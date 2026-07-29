param(
    [ValidateSet("pilot", "full")]
    [string]$Mode = "full"
)

$ErrorActionPreference = "Stop"

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutputName = if ($Mode -eq "pilot") { "access_experiment_pilot" } else { "access_experiment" }
$OutputDir = Join-Path $Repo "data_work\noto\$OutputName"
$StatusPath = Join-Path $OutputDir "run_status.json"
$PidPath = Join-Path $OutputDir "detached_run.pid"

if (-not (Test-Path -LiteralPath $StatusPath)) {
    Write-Output "No Noto $Mode status file found at $StatusPath"
    exit 0
}

$Status = Get-Content -Raw -LiteralPath $StatusPath | ConvertFrom-Json
$DetachedPid = $null
if (Test-Path -LiteralPath $PidPath) {
    $DetachedPid = [int](Get-Content -Raw -LiteralPath $PidPath)
}
$WorkerPid = if ($Status.pid) { [int]$Status.pid } else { $null }
$DetachedRunning = if ($DetachedPid) { [bool](Get-Process -Id $DetachedPid -ErrorAction SilentlyContinue) } else { $false }
$WorkerRunning = if ($WorkerPid) { [bool](Get-Process -Id $WorkerPid -ErrorAction SilentlyContinue) } else { $false }

Write-Output "Status: $($Status.status)"
Write-Output "Block: $($Status.block)"
Write-Output "Message: $($Status.message)"
Write-Output "Detached PID: $DetachedPid running=$DetachedRunning"
Write-Output "Worker PID: $WorkerPid running=$WorkerRunning"
Write-Output "Output: $($Status.output_dir)"
Write-Output "Checkpoints: $($Status.checkpoint_dir)"
Write-Output "Log: $($Status.log_path)"

if ($Status.checkpoint_dir -and (Test-Path -LiteralPath $Status.checkpoint_dir)) {
    $Checkpoints = Get-ChildItem -LiteralPath $Status.checkpoint_dir -Filter "noto-access-v1__noto_access_m5_$Mode`_rho*.json" -ErrorAction SilentlyContinue
    if ($Checkpoints.Count -gt 0) {
        $ExpectedFeasiblePerRho = if ($Mode -eq "pilot") { 83 } else { 996 }
        $TotalGridVectorsPerRho = if ($Mode -eq "pilot") { 243 } else { 3125 }
        $RhoCount = 6
        $ExpectedGridEvaluations = $ExpectedFeasiblePerRho * $RhoCount
        $GridCheckpoints = @($Checkpoints | Where-Object { $_.Name -like "*__grid_*" })
        $CompletedGridEvaluations = $GridCheckpoints.Count
        $ProgressPercent = 100.0 * $CompletedGridEvaluations / $ExpectedGridEvaluations

        Write-Output ""
        Write-Output "Noto $Mode checkpoints:"
        $Checkpoints |
            Group-Object {
                if ($_.Name -match "rho([0-9]+\.[0-9]+)") { $Matches[1] } else { "unknown" }
            } |
            Sort-Object Name |
            ForEach-Object {
                Write-Output ("  rho={0}: {1} files" -f $_.Name, $_.Count)
            }
        $Latest = $Checkpoints | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        Write-Output "  latest: $($Latest.Name) at $($Latest.LastWriteTime)"
        Write-Output ("Grid progress: {0}/{1} exact feasible evaluations ({2:N1}%)." -f $CompletedGridEvaluations, $ExpectedGridEvaluations, $ProgressPercent)
        Write-Output ("Search design: {0:N0} grid vectors per rho; {1} feasible and {2} budget-infeasible." -f $TotalGridVectorsPerRho, $ExpectedFeasiblePerRho, ($TotalGridVectorsPerRho - $ExpectedFeasiblePerRho))

        $RecentGridCheckpoints = @($GridCheckpoints | Sort-Object LastWriteTime -Descending | Select-Object -First 100)
        if ($RecentGridCheckpoints.Count -ge 2 -and $CompletedGridEvaluations -lt $ExpectedGridEvaluations) {
            $NewestTime = ($RecentGridCheckpoints | Measure-Object LastWriteTime -Maximum).Maximum
            $OldestTime = ($RecentGridCheckpoints | Measure-Object LastWriteTime -Minimum).Minimum
            $WindowMinutes = ($NewestTime - $OldestTime).TotalMinutes
            if ($WindowMinutes -gt 0) {
                $EvaluationsPerMinute = ($RecentGridCheckpoints.Count - 1) / $WindowMinutes
                $RemainingEvaluations = $ExpectedGridEvaluations - $CompletedGridEvaluations
                $Eta = [TimeSpan]::FromMinutes($RemainingEvaluations / $EvaluationsPerMinute)
                Write-Output ("Rolling throughput: {0:N1} evaluations/min; ETA {1:hh\:mm\:ss}." -f $EvaluationsPerMinute, $Eta)
            }
        }
    }
}

if ($Status.log_path -and (Test-Path -LiteralPath $Status.log_path)) {
    Write-Output ""
    Write-Output "Recent log lines:"
    Get-Content -LiteralPath $Status.log_path -Tail 25
}
