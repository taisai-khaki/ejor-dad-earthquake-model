param(
    [ValidateSet("pilot", "full")]
    [string]$Mode = "full",
    [int]$Workers = 4,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$RunScript = Join-Path $Repo "scripts\run_noto_access_experiment_resumable.ps1"
$OutputName = if ($Mode -eq "pilot") { "access_experiment_pilot" } else { "access_experiment" }
$OutputDir = Join-Path $Repo "data_work\noto\$OutputName"
$PidPath = Join-Path $OutputDir "detached_run.pid"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
if (Test-Path -LiteralPath $PidPath) {
    $ExistingPid = [int](Get-Content -Raw -LiteralPath $PidPath)
    $ExistingProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $ExistingPid" -ErrorAction SilentlyContinue
    $ExpectedRunner = "run_noto_access_experiment_resumable.ps1"
    if ($ExistingProcess -and $ExistingProcess.CommandLine -like "*$ExpectedRunner*") {
        Write-Output "Noto $Mode run is already active. PID=$ExistingPid"
        Write-Output "Check with: powershell -ExecutionPolicy Bypass -File .\scripts\status_noto_access_experiment.ps1 -Mode $Mode"
        exit 0
    }
}

$ArgumentList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $RunScript, "-Mode", $Mode, "-Workers", $Workers)
if ($Force) {
    $ArgumentList += "-Force"
}

$Process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $ArgumentList `
    -WorkingDirectory $Repo `
    -WindowStyle Hidden `
    -PassThru

$Process.Id | Set-Content -LiteralPath $PidPath -Encoding ASCII
Write-Output "Started detached Noto $Mode run. PID=$($Process.Id)"
Write-Output "Status: $OutputDir\run_status.json"
Write-Output "Check with: powershell -ExecutionPolicy Bypass -File .\scripts\status_noto_access_experiment.ps1 -Mode $Mode"
