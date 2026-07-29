param(
    [ValidateSet(2, 3)]
    [int]$NumLinks = 3,
    [double]$DensityCap = 2.0,
    [string]$RhoValues = "0,0.05,0.10,0.15,0.20,0.25",
    [int]$MaxNodes = 10000,
    [double]$TimeLimitSec = 900.0,
    [Nullable[double]]$MonotonicityAnchor = $null,
    [double]$CertificateAbsoluteTolerance = 0.1,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$RunScript = Join-Path $Repo "scripts\run_noto_support_m3_resumable.ps1"
$OutputDir = Join-Path $Repo "data_work\noto\support_preserving_m$NumLinks"
$PidPath = Join-Path $OutputDir "detached_run.pid"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
if (Test-Path -LiteralPath $PidPath) {
    $ExistingPid = [int](Get-Content -Raw -LiteralPath $PidPath)
    $ExistingProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $ExistingPid" -ErrorAction SilentlyContinue
    if ($ExistingProcess -and $ExistingProcess.CommandLine -like "*run_noto_support_m3_resumable.ps1*") {
        Write-Output "Support-preserving Noto m=$NumLinks certification is already active. PID=$ExistingPid"
        Write-Output "Check with: powershell -ExecutionPolicy Bypass -File .\scripts\status_noto_support_m3.ps1 -NumLinks $NumLinks"
        exit 0
    }
}

$ArgumentList = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $RunScript,
    "-NumLinks", $NumLinks,
    "-DensityCap", $DensityCap,
    "-RhoValues", $RhoValues,
    "-MaxNodes", $MaxNodes,
    "-TimeLimitSec", $TimeLimitSec,
    "-CertificateAbsoluteTolerance", $CertificateAbsoluteTolerance
)
if ($null -ne $MonotonicityAnchor) {
    $ArgumentList += @("-MonotonicityAnchor", $MonotonicityAnchor)
}
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
Write-Output "Started detached support-preserving Noto m=$NumLinks certification. PID=$($Process.Id)"
Write-Output "Check with: powershell -ExecutionPolicy Bypass -File .\scripts\status_noto_support_m3.ps1 -NumLinks $NumLinks"
