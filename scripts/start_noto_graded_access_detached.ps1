[CmdletBinding()]
param(
    [string]$OutputName = "acute_access_graded_v4",
    [int]$Workers = 4,
    [switch]$ForceGrid,
    [switch]$ForceCertificate
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$RunScript = Join-Path $Repo "scripts\run_noto_graded_access_resumable.ps1"
$OutputDir = Join-Path $Repo ("data_work\noto\" + $OutputName)
$PidPath = Join-Path $OutputDir "graded_runner.pid"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if (Test-Path $PidPath) {
    $existingPid = [int](Get-Content -LiteralPath $PidPath -Raw)
    if (Get-Process -Id $existingPid -ErrorAction SilentlyContinue) {
        Write-Output "Graded-response Noto run is already active. PID=$existingPid"
        exit 0
    }
}

$arguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$RunScript`"",
    "-OutputName", "`"$OutputName`"",
    "-Workers", "$Workers"
)
if ($ForceGrid) {
    $arguments += "-ForceGrid"
}
if ($ForceCertificate) {
    $arguments += "-ForceCertificate"
}
$process = Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -WorkingDirectory $Repo -WindowStyle Hidden -PassThru
Set-Content -LiteralPath $PidPath -Value $process.Id -Encoding ascii
Write-Output "Started detached graded-response Noto run. PID=$($process.Id)"
Write-Output "Status: powershell -ExecutionPolicy Bypass -File .\scripts\status_noto_graded_access.ps1 -OutputName $OutputName"

