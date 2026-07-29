[CmdletBinding()]
param(
    [string]$OutputName = "acute_access_graded_v4",
    [int]$LogLines = 20
)

$Repo = Split-Path -Parent $PSScriptRoot
$OutputDir = Join-Path $Repo ("data_work\noto\" + $OutputName)
$PidPath = Join-Path $OutputDir "graded_runner.pid"
$RunnerStatus = Join-Path $OutputDir "graded_runner_status.json"
$GridStatus = Join-Path $OutputDir "run_status.json"
$CertificateStatus = Join-Path $OutputDir "certificate_status.json"

if (Test-Path $PidPath) {
    $processId = [int](Get-Content -LiteralPath $PidPath -Raw)
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    Write-Output ("runner_pid={0}; active={1}" -f $processId, [bool]$process)
}
else {
    Write-Output "runner_pid=none; active=False"
}

foreach ($entry in @(
    @{ Name = "runner"; Path = $RunnerStatus },
    @{ Name = "grid"; Path = $GridStatus },
    @{ Name = "certificate"; Path = $CertificateStatus }
)) {
    if (Test-Path $entry.Path) {
        Write-Output ("--- {0} status ---" -f $entry.Name)
        Get-Content -LiteralPath $entry.Path
    }
}

$logDir = Join-Path $OutputDir "logs"
$latestLog = Get-ChildItem -LiteralPath $logDir -Filter "*.log" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($latestLog) {
    Write-Output ("--- latest log: {0} ---" -f $latestLog.FullName)
    Get-Content -LiteralPath $latestLog.FullName -Tail $LogLines
}

