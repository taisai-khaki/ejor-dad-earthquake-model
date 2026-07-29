$ErrorActionPreference = "Stop"

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$RunScript = Join-Path $Repo "scripts\run_turkey_paper_results_resumable.ps1"
$OutputDir = Join-Path $Repo "data_work\turkey\paper_tables"
$PidPath = Join-Path $OutputDir "detached_run.pid"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$Process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $RunScript) `
    -WorkingDirectory $Repo `
    -WindowStyle Hidden `
    -PassThru

$Process.Id | Set-Content -LiteralPath $PidPath -Encoding ASCII
Write-Output "Started detached Turkey paper-results run. PID=$($Process.Id)"
Write-Output "Status: $OutputDir\run_status.json"
Write-Output "Check with: powershell -ExecutionPolicy Bypass -File .\scripts\status_turkey_paper_results.ps1"
