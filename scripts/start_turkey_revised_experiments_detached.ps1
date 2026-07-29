$ErrorActionPreference = "Stop"

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$RunScript = Join-Path $Repo "scripts\run_turkey_revised_experiments_resumable.ps1"
$OutputDir = Join-Path $Repo "data_work\turkey\revised_experiments"
$PidPath = Join-Path $OutputDir "detached_run.pid"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$ArgumentList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $RunScript) + $args

$Process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $ArgumentList `
    -WorkingDirectory $Repo `
    -WindowStyle Hidden `
    -PassThru

$Process.Id | Set-Content -LiteralPath $PidPath -Encoding ASCII
Write-Output "Started detached Turkey revised-experiments run. PID=$($Process.Id)"
Write-Output "Status: $OutputDir\run_status.json"
Write-Output "Check with: powershell -ExecutionPolicy Bypass -File .\scripts\status_turkey_revised_experiments.ps1"
