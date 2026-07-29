$ErrorActionPreference = "Stop"

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutputDir = Join-Path $Repo "data_work\turkey\paper_tables"
$StatusPath = Join-Path $OutputDir "run_status.json"
$PidPath = Join-Path $OutputDir "detached_run.pid"

if (Test-Path -LiteralPath $StatusPath) {
    Get-Content -LiteralPath $StatusPath -Encoding UTF8
} else {
    Write-Output "No status file found at $StatusPath"
}

if (Test-Path -LiteralPath $PidPath) {
    $RunPid = [int](Get-Content -LiteralPath $PidPath -Encoding ASCII)
    $Process = Get-Process -Id $RunPid -ErrorAction SilentlyContinue
    if ($Process) {
        Write-Output "Process $RunPid is still running."
    } else {
        Write-Output "Process $RunPid is not running."
    }
}
