Write-Host "Cleaning up unnecessary files..." -ForegroundColor Yellow

Remove-Item -Recurse -Force __pycache__ -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force mlruns -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force vectorstore -ErrorAction SilentlyContinue
Remove-Item *_answer.txt -ErrorAction SilentlyContinue
Remove-Item experiment_results.json -ErrorAction SilentlyContinue
Remove-Item *.log -ErrorAction SilentlyContinue

Write-Host "Cleanup complete!" -ForegroundColor Green
Write-Host "Ready for Git commit." -ForegroundColor Cyan