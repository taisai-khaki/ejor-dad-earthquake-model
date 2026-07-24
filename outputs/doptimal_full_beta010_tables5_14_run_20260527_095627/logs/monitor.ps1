param([int]$pidTarget,[string]$outPath,[string]$logPath)
while (Get-Process -Id $pidTarget -ErrorAction SilentlyContinue) {
  $p = Get-Process -Id $pidTarget
  $files = (Get-ChildItem $outPath -Recurse -File -ErrorAction SilentlyContinue).Count
  $line = ('{0} | PID={1} | CPU={2:N2}s | WS={3:N1}MB | files={4}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $pidTarget, $p.CPU, ($p.WorkingSet64/1MB), $files)
  Add-Content -Path $logPath -Value $line
  Start-Sleep -Seconds 30
}
Add-Content -Path $logPath -Value ((Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' | PID=' + $pidTarget + ' | process exited')
