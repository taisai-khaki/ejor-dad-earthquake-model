$ErrorActionPreference = "Stop"

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutputDir = Join-Path $Repo "data_work\nepal\access_experiment"
$StatusPath = Join-Path $OutputDir "run_status.json"
$PidPath = Join-Path $OutputDir "detached_run.pid"

if (-not (Test-Path -LiteralPath $StatusPath)) {
    Write-Output "No Nepal access-experiment status file found at $StatusPath"
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
    $Checkpoints = Get-ChildItem -LiteralPath $Status.checkpoint_dir -Filter "nepal-access-v1__nepal_access_m5_rho*.json" -ErrorAction SilentlyContinue
    if ($Checkpoints.Count -gt 0) {
        Write-Output ""
        Write-Output "Nepal access checkpoints:"
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
    }
}

if ($Status.log_path -and (Test-Path -LiteralPath $Status.log_path)) {
    Write-Output ""
    Write-Output "Recent log lines:"
    Get-Content -LiteralPath $Status.log_path -Tail 25
}
