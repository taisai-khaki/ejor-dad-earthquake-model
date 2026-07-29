[CmdletBinding()]
param(
    [string]$OutputName = "acute_access_graded_v4",
    [int]$Workers = 4,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Runner = Join-Path $PSScriptRoot "run_noto_response_sensitivity_resumable.ps1"
$SensitivityDir = Join-Path $Repo ("data_work\noto\" + $OutputName + "\joint_response_sensitivity_v2")
$PidPath = Join-Path $SensitivityDir "launcher_pid.txt"
New-Item -ItemType Directory -Force -Path $SensitivityDir | Out-Null

if (Test-Path -LiteralPath $PidPath) {
    $existingPid = [int](Get-Content -LiteralPath $PidPath -Raw)
    if (Get-Process -Id $existingPid -ErrorAction SilentlyContinue) {
        throw "A response sensitivity runner is already active with PID $existingPid."
    }
}

$arguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Runner, "-OutputName", $OutputName, "-Workers", "$Workers")
if ($Force) {
    $arguments += "-Force"
}
$process = Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -WindowStyle Hidden -PassThru
Set-Content -LiteralPath $PidPath -Value $process.Id -Encoding ascii
[PSCustomObject]@{
    pid = $process.Id
    sensitivity_dir = $SensitivityDir
    status_path = Join-Path $SensitivityDir "status.json"
    runner_status_path = Join-Path $SensitivityDir "runner_status.json"
}
