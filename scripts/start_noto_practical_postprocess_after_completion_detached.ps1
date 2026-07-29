param(
    [string]$OutputDir = "",
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $Repo "data_work\noto\practical_resilience"
}
$RunScript = Join-Path $Repo "scripts\run_noto_practical_postprocess_after_completion.ps1"
$PidPath = Join-Path $OutputDir "postprocess_queue.pid"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if (Test-Path -LiteralPath $PidPath) {
    $ExistingPid = [int](Get-Content -Raw -LiteralPath $PidPath)
    $ExistingProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $ExistingPid" -ErrorAction SilentlyContinue
    if ($ExistingProcess -and $ExistingProcess.CommandLine -like "*run_noto_practical_postprocess_after_completion.ps1*") {
        Write-Output "Practical Noto postprocessor is already queued. PID=$ExistingPid"
        exit 0
    }
}

$ArgumentList = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $RunScript,
    "-OutputDir", $OutputDir,
    "-PollSeconds", $PollSeconds
)
$Process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $ArgumentList `
    -WorkingDirectory $Repo `
    -WindowStyle Hidden `
    -PassThru
$Process.Id | Set-Content -LiteralPath $PidPath -Encoding ASCII
Write-Output "Queued detached Noto post-run diagnostic. PID=$($Process.Id)"
Write-Output "Check with: powershell -ExecutionPolicy Bypass -File .\scripts\status_noto_practical_postprocess_queue.ps1"