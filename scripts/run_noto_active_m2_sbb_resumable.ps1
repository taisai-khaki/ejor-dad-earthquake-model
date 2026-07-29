param(
    [double]$BudgetMultiplier = 0.47444943359239915,
    [double]$DensityCap = 2.0,
    [string]$RhoValues = "0,0.05,0.10,0.15,0.20,0.25",
    [int]$MaxNodes = 5000,
    [double]$TimeLimitSec = 1800.0,
    [double]$CertificateAbsoluteTolerance = 0.1,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutputDir = Join-Path $Repo "data_work\noto\support_preserving_active_m2_sbb"
$LogDir = Join-Path $OutputDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ProgressLogPath = Join-Path $LogDir ("noto_active_m2_progress_" + $Timestamp + ".log")
$ProcessLogPath = Join-Path $LogDir ("noto_active_m2_process_" + $Timestamp + ".log")
$env:PYTHONPATH = "$Repo\src;$Repo\examples"
$env:EJOR_LOG_PATH = $ProgressLogPath

$Arguments = @(
    "$Repo\examples\noto_active_m2_certification.py",
    "--budget-multiplier", $BudgetMultiplier,
    "--density-cap", $DensityCap,
    "--rho-values", $RhoValues,
    "--max-nodes", $MaxNodes,
    "--time-limit-sec", $TimeLimitSec,
    "--certificate-absolute-tolerance", $CertificateAbsoluteTolerance
)
if ($Force) {
    $Arguments += "--force"
}
python @Arguments *>> $ProcessLogPath
exit $LASTEXITCODE
