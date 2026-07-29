param(
    [ValidateSet("pilot", "full")]
    [string]$Mode = "pilot"
)

$ErrorActionPreference = "Stop"

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutputName = if ($Mode -eq "pilot") { "support_preserving_pilot" } else { "support_preserving_full" }
$OutputDir = Join-Path $Repo "data_work\noto\$OutputName"
$StatusPath = Join-Path $OutputDir "run_status.json"
$PidPath = Join-Path $OutputDir "detached_run.pid"
$DesignPath = Join-Path $OutputDir "run_design.json"

if (-not (Test-Path -LiteralPath $StatusPath)) {
    Write-Output "No support-preserving Noto $Mode status file found at $StatusPath"
    exit 0
}

$Status = Get-Content -Raw -LiteralPath $StatusPath | ConvertFrom-Json
$DetachedPid = if (Test-Path -LiteralPath $PidPath) { [int](Get-Content -Raw -LiteralPath $PidPath) } else { $null }
$WorkerPid = if ($Status.pid) { [int]$Status.pid } else { $null }
$DetachedRunning = if ($DetachedPid) { [bool](Get-Process -Id $DetachedPid -ErrorAction SilentlyContinue) } else { $false }
$WorkerRunning = if ($WorkerPid) { [bool](Get-Process -Id $WorkerPid -ErrorAction SilentlyContinue) } else { $false }
$Design = if (Test-Path -LiteralPath $DesignPath) { Get-Content -Raw -LiteralPath $DesignPath | ConvertFrom-Json } else { $null }

Write-Output "Status: $($Status.status)"
Write-Output "Message: $($Status.message)"
Write-Output "Detached PID: $DetachedPid running=$DetachedRunning"
Write-Output "Worker PID: $WorkerPid running=$WorkerRunning"
Write-Output "Output: $OutputDir"
Write-Output "Log: $($Status.log_path)"

$CheckpointDir = Join-Path $OutputDir "checkpoints"
if (Test-Path -LiteralPath $CheckpointDir) {
    $GridCheckpoints = @(Get-ChildItem -LiteralPath $CheckpointDir -Filter "*noto_support_m5_$Mode`_k*_rho*__grid_*.json" -File)
    if ($GridCheckpoints.Count -gt 0) {
        Write-Output ""
        Write-Output "Grid checkpoints: $($GridCheckpoints.Count)"
        $GridCheckpoints |
            Group-Object {
                if ($_.Name -match "_k([^_]+)_rho([0-9]+\.[0-9]+)") { "k=$($Matches[1]) rho=$($Matches[2])" } else { "unknown" }
            } |
            Sort-Object Name |
            ForEach-Object { Write-Output ("  {0}: {1}" -f $_.Name, $_.Count) }
        $Latest = $GridCheckpoints | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        Write-Output "Latest: $($Latest.Name) at $($Latest.LastWriteTime)"

        if ($Design) {
            $Expected = [int]$Design.expected_exact_evaluations
            $Completed = $GridCheckpoints.Count
            $ProgressPercent = if ($Expected -gt 0) { 100.0 * $Completed / $Expected } else { 0.0 }
            Write-Output ("Grid progress: {0}/{1} exact feasible evaluations ({2:N1}%)." -f $Completed, $Expected, $ProgressPercent)
            Write-Output ("Search design: {0} cap(s), {1} radius values, {2:N0} grid vectors and {3:N0} feasible policies per scenario." -f $Design.density_caps.Count, $Design.rho_values.Count, $Design.total_grid_vectors_per_scenario, $Design.feasible_grid_vectors_per_scenario)

            $Recent = @($GridCheckpoints | Sort-Object LastWriteTime -Descending | Select-Object -First 100)
            if ($Recent.Count -ge 2 -and $Completed -lt $Expected) {
                $NewestTime = ($Recent | Measure-Object LastWriteTime -Maximum).Maximum
                $OldestTime = ($Recent | Measure-Object LastWriteTime -Minimum).Minimum
                $WindowMinutes = ($NewestTime - $OldestTime).TotalMinutes
                if ($WindowMinutes -gt 0) {
                    $Rate = ($Recent.Count - 1) / $WindowMinutes
                    $Eta = [TimeSpan]::FromMinutes(($Expected - $Completed) / $Rate)
                    Write-Output ("Rolling throughput: {0:N1} evaluations/min; ETA {1:hh\:mm\:ss}." -f $Rate, $Eta)
                }
            }
        }
    }
}

if ($Status.log_path -and (Test-Path -LiteralPath $Status.log_path)) {
    Write-Output ""
    Write-Output "Recent log lines:"
    Get-Content -LiteralPath $Status.log_path -Tail 25
}
