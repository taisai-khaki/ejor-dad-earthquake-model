param(
    [ValidateSet("pilot", "full")]
    [string]$Mode = "full",
    [double]$DensityCap = 2.0,
    [double]$ResidualFailureRatio = 0.10,
    [double]$FailureDelayReduction = 0.50,
    [double]$TimeSensitiveFraction = 1.0,
    [double]$ImmediateLossFraction = 0.0,
    [Nullable[double]]$CapacityThroughputPerBed = $null,
    [Nullable[double]]$ResponseThresholdMinutes = $null,
    [double]$RetrofitBudgetScale = 1.0,
    [string]$RhoValues = "0,0.05,0.10,0.15,0.20,0.25",
    [int]$Workers = 4,
    [string]$OutputDir = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $Repo "data_work\noto\practical_resilience"
}
$LogDir = Join-Path $OutputDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ProgressLogPath = Join-Path $LogDir ("noto_practical_progress_" + $Timestamp + ".log")
$ProcessLogPath = Join-Path $LogDir ("noto_practical_process_" + $Timestamp + ".log")
$env:PYTHONPATH = "$Repo\src;$Repo\examples"
$env:EJOR_LOG_PATH = $ProgressLogPath

$Arguments = @(
    "$Repo\examples\noto_practical_resilience_experiment.py",
    "--mode", $Mode,
    "--density-cap", $DensityCap,
    "--residual-failure-ratio", $ResidualFailureRatio,
    "--failure-delay-reduction", $FailureDelayReduction,
    "--time-sensitive-fraction", $TimeSensitiveFraction,
    "--immediate-loss-fraction", $ImmediateLossFraction,
    "--retrofit-budget-scale", $RetrofitBudgetScale,
    "--rho-values", $RhoValues,
    "--workers", $Workers,
    "--output-dir", $OutputDir
)
if ($null -ne $CapacityThroughputPerBed) {
    $Arguments += @("--capacity-throughput-per-bed", $CapacityThroughputPerBed)
}
if ($null -ne $ResponseThresholdMinutes) {
    $Arguments += @("--response-threshold-minutes", $ResponseThresholdMinutes)
}
if ($Force) {
    $Arguments += "--force"
}

python @Arguments *>> $ProcessLogPath
exit $LASTEXITCODE