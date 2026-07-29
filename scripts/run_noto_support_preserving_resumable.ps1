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
$OutputName = if ($Mode -eq "pilot") { "support_preserving_pilot" } else { "support_preserving_full" }
$OutputDir = Join-Path $Repo "data_work\noto\$OutputName"
$LogDir = Join-Path $OutputDir "logs"
$StatusPath = Join-Path $OutputDir "run_status.json"
$LogPath = Join-Path $LogDir ("noto_support_{0}_{1:yyyyMMdd_HHmmss}.log" -f $Mode, (Get-Date))

New-Item -ItemType Directory -Force -Path $OutputDir, $LogDir | Out-Null

function Write-Status {
    param(
        [string]$Status,
        [int]$ExitCode = 0,
        [string]$Message = ""
    )
    @{
        status = $Status
        exit_code = $ExitCode
        message = $Message
        pid = $PID
        updated_at = (Get-Date).ToString("o")
        log_path = $LogPath
        output_dir = $OutputDir
        checkpoint_dir = (Join-Path $OutputDir "checkpoints")
        mode = $Mode
        density_caps = $DensityCaps
        rho_values = $RhoValues
        workers = $Workers
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

Set-Location $Repo
$env:PYTHONPATH = "$Repo\src"
$env:PYTHONIOENCODING = "utf-8"
$env:EJOR_LOG_PATH = $LogPath
$PythonArgs = @(
    ".\examples\noto_support_preserving_experiment.py",
    "--mode", $Mode,
    "--density-caps", $DensityCaps,
    "--rho-values", $RhoValues,
    "--workers", $Workers
)
if ($Force) {
    $PythonArgs += "--force"
}
Write-Status -Status "running" -Message "Support-preserving Noto $Mode sweep started."

try {
    python @PythonArgs 2>&1 | Tee-Object -FilePath $LogPath -Append
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -eq 0) {
        Write-Status -Status "completed" -ExitCode 0 -Message "Support-preserving Noto $Mode sweep completed."
    } else {
        Write-Status -Status "failed" -ExitCode $ExitCode -Message "Python exited with a nonzero status."
    }
    exit $ExitCode
} catch {
    Write-Status -Status "failed" -ExitCode 1 -Message $_.Exception.Message
    throw
}
