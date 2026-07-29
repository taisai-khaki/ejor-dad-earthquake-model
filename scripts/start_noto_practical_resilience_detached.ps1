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
$RunScript = Join-Path $Repo "scripts\run_noto_practical_resilience_resumable.ps1"
$PidPath = Join-Path $OutputDir "detached_run.pid"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if (Test-Path -LiteralPath $PidPath) {
    $ExistingPid = [int](Get-Content -Raw -LiteralPath $PidPath)
    $ExistingProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $ExistingPid" -ErrorAction SilentlyContinue
    if ($ExistingProcess -and $ExistingProcess.CommandLine -like "*run_noto_practical_resilience_resumable.ps1*") {
        Write-Output "Practical Noto analysis is already active. PID=$ExistingPid"
        exit 0
    }
}

$ArgumentList = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $RunScript,
    "-Mode", $Mode,
    "-DensityCap", $DensityCap,
    "-ResidualFailureRatio", $ResidualFailureRatio,
    "-FailureDelayReduction", $FailureDelayReduction,
    "-TimeSensitiveFraction", $TimeSensitiveFraction,
    "-ImmediateLossFraction", $ImmediateLossFraction,
    "-RetrofitBudgetScale", $RetrofitBudgetScale,
    "-RhoValues", $RhoValues,
    "-Workers", $Workers,
    "-OutputDir", $OutputDir
)
if ($null -ne $CapacityThroughputPerBed) {
    $ArgumentList += @("-CapacityThroughputPerBed", $CapacityThroughputPerBed)
}
if ($null -ne $ResponseThresholdMinutes) {
    $ArgumentList += @("-ResponseThresholdMinutes", $ResponseThresholdMinutes)
}
if ($Force) {
    $ArgumentList += "-Force"
}

$Process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $ArgumentList `
    -WorkingDirectory $Repo `
    -WindowStyle Hidden `
    -PassThru

$Process.Id | Set-Content -LiteralPath $PidPath -Encoding ASCII
Write-Output "Started detached practical Noto analysis. PID=$($Process.Id)"
Write-Output "Check with: powershell -ExecutionPolicy Bypass -File .\scripts\status_noto_practical_resilience.ps1"