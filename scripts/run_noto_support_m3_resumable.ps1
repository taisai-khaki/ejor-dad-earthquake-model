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
$OutputDir = Join-Path $Repo "data_work\noto\support_preserving_m$NumLinks"
$LogDir = Join-Path $OutputDir "logs"
$StatusPath = Join-Path $OutputDir "run_status.json"
$LogPath = Join-Path $LogDir ("noto_support_m{0}_{1:yyyyMMdd_HHmmss}.log" -f $NumLinks, (Get-Date))

New-Item -ItemType Directory -Force -Path $OutputDir, $LogDir | Out-Null

function Write-Status {
    param([string]$Status, [int]$ExitCode = 0, [string]$Message = "")
    @{
        status = $Status
        exit_code = $ExitCode
        message = $Message
        pid = $PID
        updated_at = (Get-Date).ToString("o")
        log_path = $LogPath
        output_dir = $OutputDir
        checkpoint_dir = (Join-Path $OutputDir "checkpoints")
        num_links = $NumLinks
        density_cap = $DensityCap
        rho_values = $RhoValues
        max_nodes = $MaxNodes
        time_limit_sec = $TimeLimitSec
        monotonicity_anchor = $MonotonicityAnchor
        certificate_absolute_tolerance = $CertificateAbsoluteTolerance
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

Set-Location $Repo
$env:PYTHONPATH = "$Repo\src"
$env:PYTHONIOENCODING = "utf-8"
$env:EJOR_LOG_PATH = $LogPath
$PythonArgs = @(
    ".\examples\noto_support_m3_certification.py",
    "--num-links", $NumLinks,
    "--density-cap", $DensityCap,
    "--rho-values", $RhoValues,
    "--max-nodes", $MaxNodes,
    "--time-limit-sec", $TimeLimitSec,
    "--certificate-absolute-tolerance", $CertificateAbsoluteTolerance
)
if ($null -ne $MonotonicityAnchor) {
    $PythonArgs += @("--monotonicity-anchor", $MonotonicityAnchor)
}
if ($Force) {
    $PythonArgs += "--force"
}
Write-Status -Status "running" -Message "Support-preserving Noto m=$NumLinks certification started."

try {
    python @PythonArgs 2>&1 | Tee-Object -FilePath $LogPath -Append
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -eq 0) {
        Write-Status -Status "completed" -ExitCode 0 -Message "Support-preserving Noto m=$NumLinks certification completed."
    } else {
        Write-Status -Status "failed" -ExitCode $ExitCode -Message "Python exited with a nonzero status."
    }
    exit $ExitCode
} catch {
    Write-Status -Status "failed" -ExitCode 1 -Message $_.Exception.Message
    throw
}
