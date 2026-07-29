$ErrorActionPreference = "Stop"

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutputDir = Join-Path $Repo "data_work\turkey\paper_tables"
$LogDir = Join-Path $Repo "data_work\turkey\logs"
$StatusPath = Join-Path $OutputDir "run_status.json"
$LogPath = Join-Path $LogDir ("turkey_paper_results_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))

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
        checkpoint_dir = (Join-Path $OutputDir "checkpoints")
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

Set-Location $Repo
$env:PYTHONPATH = "$Repo\src;$Repo\examples"
Write-Status -Status "running" -Message "Turkey paper result generation started."

try {
    python .\examples\turkey_paper_results.py @args 2>&1 | Tee-Object -FilePath $LogPath -Append
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -eq 0) {
        Write-Status -Status "completed" -ExitCode 0 -Message "Turkey paper result generation completed."
    } else {
        Write-Status -Status "failed" -ExitCode $ExitCode -Message "Python exited with a nonzero status."
    }
    exit $ExitCode
} catch {
    Write-Status -Status "failed" -ExitCode 1 -Message $_.Exception.Message
    throw
}
