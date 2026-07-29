param(
    [double]$BudgetMultiplier = 0.47444943359239915,
    [double]$GridStep = 0.05,
    [double]$DensityCap = 2.0,
    [string]$RhoValues = "0,0.05,0.10,0.15,0.20,0.25",
    [int]$Workers = 4,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$RunScript = Join-Path $Repo "scripts\run_noto_active_fine_grid_resumable.ps1"
$OutputDir = Join-Path $Repo "data_work\noto\support_preserving_active_fine_grid"
$PidPath = Join-Path $OutputDir "detached_run.pid"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if (Test-Path -LiteralPath $PidPath) {
    $ExistingPid = [int](Get-Content -Raw -LiteralPath $PidPath)
    $ExistingProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $ExistingPid" -ErrorAction SilentlyContinue
    if ($ExistingProcess -and $ExistingProcess.CommandLine -like "*run_noto_active_fine_grid_resumable.ps1*") {
        Write-Output "Noto active fine-grid analysis is already active. PID=$ExistingPid"
        exit 0
    }
}

$ArgumentList = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $RunScript,
    "-BudgetMultiplier", $BudgetMultiplier,
    "-GridStep", $GridStep,
    "-DensityCap", $DensityCap,
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
Write-Output "Started detached Noto active fine-grid analysis. PID=$($Process.Id)"
Write-Output "Check with: powershell -ExecutionPolicy Bypass -File .\scripts\status_noto_active_fine_grid.ps1"
