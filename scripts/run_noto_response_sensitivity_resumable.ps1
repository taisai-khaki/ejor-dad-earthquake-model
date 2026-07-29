[CmdletBinding()]
param(
    [string]$OutputName = "acute_access_graded_v4",
    [int]$Workers = 4,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$BaseOutputDir = Join-Path $Repo ("data_work\noto\" + $OutputName)
$SensitivityDir = Join-Path $BaseOutputDir "joint_response_sensitivity_v2"
$LogDir = Join-Path $SensitivityDir "logs"
$RunnerStatus = Join-Path $SensitivityDir "runner_status.json"
$RunLog = Join-Path $LogDir ("response_sensitivity_runner_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$env:PYTHONPATH = Join-Path $Repo "src"
$env:PYTHONDONTWRITEBYTECODE = "1"

function Write-RunnerStatus {
    param(
        [string]$Status,
        [string]$Message,
        [int]$ExitCode = -1
    )
    $payload = [ordered]@{
        status = $Status
        message = $Message
        pid = $PID
        updated_at = (Get-Date).ToUniversalTime().ToString("o")
        base_output_dir = $BaseOutputDir
        sensitivity_dir = $SensitivityDir
        log_path = $RunLog
        workers = $Workers
    }
    if ($ExitCode -ge 0) {
        $payload.exit_code = $ExitCode
    }
    $temporary = "$RunnerStatus.tmp-$PID"
    $payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $RunnerStatus -Force
}

try {
    Write-RunnerStatus -Status "running" -Message "Restart-safe graded-response sensitivity enumeration started."
    Set-Location $Repo
    $arguments = @(
        ".\examples\noto_joint_sensitivity.py",
        "--base-output-dir", $BaseOutputDir,
        "--output-dir", $SensitivityDir,
        "--only-response",
        "--workers", "$Workers"
    )
    if ($Force) {
        $arguments += "--force"
    }
    & python @arguments 2>&1 | Tee-Object -FilePath $RunLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Response sensitivity runner failed with exit code $LASTEXITCODE."
    }
    Write-RunnerStatus -Status "completed" -Message "Restart-safe graded-response sensitivity enumeration completed." -ExitCode 0
}
catch {
    Write-RunnerStatus -Status "failed" -Message $_.Exception.Message -ExitCode 1
    throw
}
