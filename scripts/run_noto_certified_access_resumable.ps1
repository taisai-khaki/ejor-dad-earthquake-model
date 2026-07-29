[CmdletBinding()]
param(
    [string]$OutputName = "acute_access_certified_v3",
    [int]$Workers = 4,
    [switch]$ForceGrid,
    [switch]$ForceCertificate
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$OutputDir = Join-Path $Repo ("data_work\noto\" + $OutputName)
$LogDir = Join-Path $OutputDir "logs"
$RunLog = Join-Path $LogDir ("noto_certified_runner_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))
$RunnerStatus = Join-Path $OutputDir "certified_runner_status.json"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$env:PYTHONPATH = Join-Path $Repo "src"
$env:PYTHONDONTWRITEBYTECODE = "1"

function Write-RunnerStatus {
    param(
        [string]$Status,
        [string]$Block,
        [string]$Message,
        [int]$ExitCode = -1
    )
    $payload = [ordered]@{
        status = $Status
        block = $Block
        message = $Message
        pid = $PID
        updated_at = (Get-Date).ToUniversalTime().ToString("o")
        output_dir = $OutputDir
        log_path = $RunLog
    }
    if ($ExitCode -ge 0) {
        $payload.exit_code = $ExitCode
    }
    $temp = "$RunnerStatus.tmp-$PID"
    $payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $temp -Encoding utf8
    Move-Item -LiteralPath $temp -Destination $RunnerStatus -Force
}

try {
    Write-RunnerStatus -Status "running" -Block "grid" -Message "Corrected full Noto grid sweep started."
    $gridArgs = @(
        ".\examples\noto_practical_resilience_experiment.py",
        "--mode", "full",
        "--rho-values", "0,0.05,0.10,0.15,0.20,0.25",
        "--density-cap", "2.0",
        "--residual-failure-ratio", "0.10",
        "--failure-delay-reduction", "0.50",
        "--time-sensitive-fraction", "0.25",
        "--immediate-loss-fraction", "0.0",
        "--capacity-throughput-per-bed", "1.0",
        "--response-threshold-minutes", "60",
        "--workers", "$Workers",
        "--output-dir", $OutputDir
    )
    if ($ForceGrid) {
        $gridArgs += "--force"
    }
    & python @gridArgs 2>&1 | Tee-Object -FilePath $RunLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "The corrected Noto grid sweep failed with exit code $LASTEXITCODE."
    }

    Write-RunnerStatus -Status "running" -Block "continuous_certificate" -Message "Exact grid complete; continuous certificate started."
    $certificateArgs = @(
        ".\examples\noto_certified_postprocess.py",
        "--output-dir", $OutputDir,
        "--workers", "$Workers"
    )
    if ($ForceCertificate) {
        $certificateArgs += "--force"
    }
    & python @certificateArgs 2>&1 | Tee-Object -FilePath $RunLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "The continuous certificate postprocess failed with exit code $LASTEXITCODE."
    }

    Write-RunnerStatus -Status "completed" -Block "complete" -Message "Corrected grid, certificate, and diagnostics completed." -ExitCode 0
}
catch {
    Write-RunnerStatus -Status "failed" -Block "error" -Message $_.Exception.Message -ExitCode 1
    throw
}
