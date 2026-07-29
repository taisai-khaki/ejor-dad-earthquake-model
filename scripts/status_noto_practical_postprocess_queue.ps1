param(
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $Repo "data_work\noto\practical_resilience"
}
$QueueStatusPath = Join-Path $OutputDir "postprocess_queue_status.json"
$PostprocessStatusPath = Join-Path $OutputDir "postprocess_status.json"
$PidPath = Join-Path $OutputDir "postprocess_queue.pid"

if (-not (Test-Path -LiteralPath $QueueStatusPath)) {
    Write-Output "No queued practical Noto postprocessor status found at $QueueStatusPath"
    exit 0
}

$QueueStatus = Get-Content -Raw -LiteralPath $QueueStatusPath | ConvertFrom-Json
$DetachedPid = if (Test-Path -LiteralPath $PidPath) { [int](Get-Content -Raw -LiteralPath $PidPath) } else { $null }
$DetachedRunning = if ($DetachedPid) { [bool](Get-Process -Id $DetachedPid -ErrorAction SilentlyContinue) } else { $false }
Write-Output "Queue status: $($QueueStatus.status)"
Write-Output "Message: $($QueueStatus.message)"
Write-Output "Queue PID: $DetachedPid running=$DetachedRunning"
Write-Output "Queue log: $($QueueStatus.queue_log_path)"

if (Test-Path -LiteralPath $PostprocessStatusPath) {
    $PostprocessStatus = Get-Content -Raw -LiteralPath $PostprocessStatusPath | ConvertFrom-Json
    Write-Output "Postprocess status: $($PostprocessStatus.status)"
    Write-Output "Postprocess message: $($PostprocessStatus.message)"
}

$RecommendationPath = Join-Path $OutputDir "follow_up_recommendation.json"
if (Test-Path -LiteralPath $RecommendationPath) {
    $Recommendation = Get-Content -Raw -LiteralPath $RecommendationPath | ConvertFrom-Json
    Write-Output ""
    Write-Output "Recommendation: $($Recommendation.primary_recommendation)"
    $Recommendation.actions | ForEach-Object { Write-Output "- $_" }
}

if (Test-Path -LiteralPath $QueueStatus.queue_log_path) {
    Write-Output "Recent queue log lines:"
    Get-Content -LiteralPath $QueueStatus.queue_log_path -Tail 20
}