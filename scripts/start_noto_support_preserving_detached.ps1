param(
    [ValidateSet("pilot", "full")]
    [string]$Mode = "pilot",
    [string]$DensityCaps = "1.5,2,5,10",
    [string]$RhoValues = "0,0.05,0.10,0.15,0.20,0.25",
    [int]$Workers = 4,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$RunScript = Join-Path $Repo "scripts\run_noto_support_preserving_resumable.ps1"
$OutputName = if ($Mode -eq "pilot") { "support_preserving_pilot" } else { "support_preserving_full" }
$OutputDir = Join-Path $Repo "data_work\noto\$OutputName"
$PidPath = Join-Path $OutputDir "detached_run.pid"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
if (Test-Path -LiteralPath $PidPath) {
    $ExistingPid = [int](Get-Content -Raw -LiteralPath $PidPath)
    $ExistingProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $ExistingPid" -ErrorAction SilentlyContinue
    if ($ExistingProcess -and $ExistingProcess.CommandLine -like "*run_noto_support_preserving_resumable.ps1*") {
        Write-Output "Support-preserving Noto $Mode run is already active. PID=$ExistingPid"
        Write-Output "Check with: powershell -ExecutionPolicy Bypass -File .\scripts\status_noto_support_preserving.ps1 -Mode $Mode"
        exit 0
    }
}

$ArgumentList = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $RunScript,
    "-Mode", $Mode,
    "-DensityCaps", $DensityCaps,
    "-RhoValues", $RhoValues,
    "-Workers", $Workers
)
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
Write-Output "Started detached support-preserving Noto $Mode run. PID=$($Process.Id)"
Write-Output "Check with: powershell -ExecutionPolicy Bypass -File .\scripts\status_noto_support_preserving.ps1 -Mode $Mode"
