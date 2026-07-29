param(
    [ValidateSet("pilot", "full")]
    [string]$Mode = "full",
    [int]$Workers = 4,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutputName = if ($Mode -eq "pilot") { "access_experiment_pilot" } else { "access_experiment" }
$OutputDir = Join-Path $Repo "data_work\noto\$OutputName"
$LogDir = Join-Path $OutputDir "logs"
$StatusPath = Join-Path $OutputDir "run_status.json"
$LogPath = Join-Path $LogDir ("noto_access_{0}_{1:yyyyMMdd_HHmmss}.log" -f $Mode, (Get-Date))

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
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

Set-Location $Repo
$env:PYTHONPATH = "$Repo\src"
$env:PYTHONIOENCODING = "utf-8"
$env:EJOR_LOG_PATH = $LogPath
$PythonArgs = @(".\examples\noto_access_experiment.py", "--mode", $Mode, "--workers", $Workers)
if ($Force) {
    $PythonArgs += "--force"
}
Write-Status -Status "running" -Message "Noto $Mode experiment started."

try {
    python @PythonArgs 2>&1 | Tee-Object -FilePath $LogPath -Append
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -eq 0) {
        Write-Status -Status "running" -Message "Noto $Mode experiment completed; creating paper tables."
        python .\examples\noto_postprocess_results.py --mode $Mode --paper-rho 0.10 2>&1 | Tee-Object -FilePath $LogPath -Append
        $ExitCode = $LASTEXITCODE
    }
    if ($ExitCode -eq 0) {
        Write-Status -Status "running" -Message "Noto $Mode paper tables completed; creating figures."
        python .\examples\noto_make_figures.py --mode $Mode --paper-rho 0.10 2>&1 | Tee-Object -FilePath $LogPath -Append
        $ExitCode = $LASTEXITCODE
    }
    if ($ExitCode -eq 0) {
        Write-Status -Status "completed" -ExitCode 0 -Message "Noto $Mode experiment, paper tables, and figures completed."
    } else {
        Write-Status -Status "failed" -ExitCode $ExitCode -Message "Python exited with a nonzero status."
    }
    exit $ExitCode
} catch {
    Write-Status -Status "failed" -ExitCode 1 -Message $_.Exception.Message
    throw
}
