Write-Host "Starting RAG Multi-LLM Comparison System..." -ForegroundColor Green

# Check if Ollama is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -ErrorAction Stop
    Write-Host "✓ Ollama is running" -ForegroundColor Green
} catch {
    Write-Host "✗ Error: Ollama is not running. Please start Ollama first." -ForegroundColor Red
    exit 1
}

# Build containers
Write-Host "`nBuilding Docker containers..." -ForegroundColor Yellow
docker-compose build

# Start services
Write-Host "`nStarting services..." -ForegroundColor Yellow
docker-compose up -d

# Wait for services
Write-Host "`nWaiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Health check
Write-Host "`nService Status:" -ForegroundColor Cyan
Write-Host "===============" -ForegroundColor Cyan

try {
    Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -ErrorAction Stop | Out-Null
    Write-Host "✓ FastAPI: http://localhost:8000" -ForegroundColor Green
} catch {
    Write-Host "✗ FastAPI: Not ready" -ForegroundColor Red
}

try {
    Invoke-WebRequest -Uri "http://localhost:7860" -UseBasicParsing -ErrorAction Stop | Out-Null
    Write-Host "✓ Gradio UI: http://localhost:7860" -ForegroundColor Green
} catch {
    Write-Host "✗ Gradio UI: Not ready" -ForegroundColor Red
}

try {
    Invoke-WebRequest -Uri "http://localhost:5000" -UseBasicParsing -ErrorAction Stop | Out-Null
    Write-Host "✓ MLflow: http://localhost:5000" -ForegroundColor Green
} catch {
    Write-Host "✗ MLflow: Not ready" -ForegroundColor Red
}

try {
    Invoke-WebRequest -Uri "http://localhost:9090" -UseBasicParsing -ErrorAction Stop | Out-Null
    Write-Host "✓ Prometheus: http://localhost:9090" -ForegroundColor Green
} catch {
    Write-Host "✗ Prometheus: Not ready" -ForegroundColor Red
}

try {
    Invoke-WebRequest -Uri "http://localhost:3000" -UseBasicParsing -ErrorAction Stop | Out-Null
    Write-Host "✓ Grafana: http://localhost:3000 (admin/admin)" -ForegroundColor Green
} catch {
    Write-Host "✗ Grafana: Not ready" -ForegroundColor Red
}

Write-Host "`nAll services started!" -ForegroundColor Green
Write-Host "`nView logs: docker-compose logs -f" -ForegroundColor Yellow
Write-Host "Stop all: docker-compose down" -ForegroundColor Yellow