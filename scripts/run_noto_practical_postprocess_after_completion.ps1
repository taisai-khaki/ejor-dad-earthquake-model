param(
    [string]$OutputDir = "",
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
if ($PollSeconds -lt 5) {
    throw "PollSeconds must be at least 5."
}
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $Repo "data_work\noto\practical_resilience"
}
$LogDir = Join-Path $OutputDir "logs"
$QueueLog = Join-Path $LogDir "noto_practical_postprocess_queue.log"
$QueueStatusPath = Join-Path $OutputDir "postprocess_queue_status.json"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-QueueStatus {
    param([string]$Status, [string]$Message)
    $Payload = [ordered]@{
        status = $Status
        message = $Message
        pid = $PID
        updated_at_epoch = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
        output_dir = (Resolve-Path $OutputDir).Path
        queue_log_path = $QueueLog
    }
    $Payload | ConvertTo-Json | Set-Content -LiteralPath $QueueStatusPath -Encoding UTF8
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message" | Add-Content -LiteralPath $QueueLog
}

$RunStatusPath = Join-Path $OutputDir "run_status.json"
Write-QueueStatus "waiting" "Waiting for the all-rho practical Noto run to complete."
while ($true) {
    if (Test-Path -LiteralPath $RunStatusPath) {
        $RunStatus = Get-Content -Raw -LiteralPath $RunStatusPath | ConvertFrom-Json
        if ($RunStatus.status -eq "completed") {
            break
        }
        if ($RunStatus.status -eq "failed") {
            Write-QueueStatus "blocked" "Practical Noto run failed; postprocessing was not started."
            exit 1
        }
        Write-QueueStatus "waiting" "Practical Noto status=$($RunStatus.status), block=$($RunStatus.block); waiting."
    }
    Start-Sleep -Seconds $PollSeconds
}

$env:PYTHONPATH = "$Repo\src;$Repo\examples"
$Arguments = @("$Repo\examples\noto_practical_postprocess.py", "--output-dir", $OutputDir)
$CompletePolicyArguments = @("$Repo\examples\noto_complete_policy_diagnostic.py", "--output-dir", $OutputDir)
Write-QueueStatus "running" "All-rho run completed; generating post-run decision tables and memo."
python @Arguments *>> $QueueLog
if ($LASTEXITCODE -ne 0) {
    Write-QueueStatus "failed" "Post-run diagnostic failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}
python @CompletePolicyArguments *>> $QueueLog
if ($LASTEXITCODE -ne 0) {
    Write-QueueStatus "failed" "Complete-policy diagnostic failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}
Write-QueueStatus "completed" "Saved post-run tables, memo, and complete-policy adjustment diagnostic."
