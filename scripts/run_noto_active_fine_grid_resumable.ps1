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
$OutputDir = Join-Path $Repo "data_work\noto\support_preserving_active_fine_grid"
$LogDir = Join-Path $OutputDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ProgressLogPath = Join-Path $LogDir ("noto_active_fine_progress_" + $Timestamp + ".log")
$ProcessLogPath = Join-Path $LogDir ("noto_active_fine_process_" + $Timestamp + ".log")
$env:PYTHONPATH = "$Repo\src;$Repo\examples"
$env:EJOR_LOG_PATH = $ProgressLogPath

$Arguments = @(
    "$Repo\examples\noto_active_fine_grid.py",
    "--budget-multiplier", $BudgetMultiplier,
    "--grid-step", $GridStep,
    "--density-cap", $DensityCap,
    "--rho-values", $RhoValues,
    "--workers", $Workers
)
if ($Force) {
    $Arguments += "--force"
}

python @Arguments *>> $ProcessLogPath
exit $LASTEXITCODE
