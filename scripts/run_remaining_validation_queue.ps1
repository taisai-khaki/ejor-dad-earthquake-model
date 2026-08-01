$ErrorActionPreference = 'Stop'
$Project = 'C:\Users\L03128674\projects\ejor_dad_nepal'
$Root = Join-Path $Project 'data_work\noto\acute_access_graded_v4\correlated_facility_separated_capability_marginal_v1'
$DensePid = [int](Get-Content (Join-Path $Root 'dense_launcher_pid.txt'))
$Log = Join-Path $Root 'logs\remaining_validation_queue.log'
while (Get-Process -Id $DensePid -ErrorAction SilentlyContinue) {
  Add-Content -LiteralPath $Log -Value "$(Get-Date -Format s) waiting for dense full-grid PID $DensePid"
  Start-Sleep -Seconds 120
}
Set-Location $Project
$env:PYTHONPATH = (Resolve-Path 'src').Path
Add-Content -LiteralPath $Log -Value "$(Get-Date -Format s) dense run ended; starting dense validation postprocess"
python examples\noto_correlated_validation_postprocess.py --output-dir data_work\noto\acute_access_graded_v4 *>> $Log
if ($LASTEXITCODE -ne 0) { throw "Dense validation postprocess failed with exit code $LASTEXITCODE" }
python examples\noto_dense_full_grid_finalize.py *>> $Log
if ($LASTEXITCODE -ne 0) { throw "Dense table finalization failed with exit code $LASTEXITCODE" }
Add-Content -LiteralPath $Log -Value "$(Get-Date -Format s) dense validation package completed"
