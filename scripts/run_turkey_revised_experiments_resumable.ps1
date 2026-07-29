$ErrorActionPreference = "Stop"

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutputDir = Join-Path $Repo "data_work\turkey\revised_experiments"
$LogDir = Join-Path $OutputDir "logs"
$StatusPath = Join-Path $OutputDir "run_status.json"
$LogPath = Join-Path $LogDir ("turkey_revised_experiments_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))

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
        output_dir = $OutputDir
        checkpoint_dir = (Join-Path $OutputDir "checkpoints")
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

Set-Location $Repo
$env:PYTHONPATH = "$Repo\src;$Repo\examples"
Write-Status -Status "running" -Message "Turkey revised experiments started."

try {
    python .\examples\turkey_revised_experiments.py @args 2>&1 | Tee-Object -FilePath $LogPath -Append
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -eq 0) {
        Write-Status -Status "completed" -ExitCode 0 -Message "Turkey revised experiments completed."
    } else {
        Write-Status -Status "failed" -ExitCode $ExitCode -Message "Python exited with a nonzero status."
    }
    exit $ExitCode
} catch {
    Write-Status -Status "failed" -ExitCode 1 -Message $_.Exception.Message
    throw
}
