#!/bin/bash

echo "Starting RAG Multi-LLM Comparison System..."

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Error: Ollama is not running. Please start Ollama first."
    exit 1
fi

# Build and start containers
echo "Building Docker containers..."
docker-compose build

echo "Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Health check
echo ""
echo "Service Status:"
echo "==============="
curl -s http://localhost:8000/health && echo "✓ FastAPI: http://localhost:8000"
curl -s http://localhost:7860 > /dev/null && echo "✓ Gradio UI: http://localhost:7860"
curl -s http://localhost:5000 > /dev/null && echo "✓ MLflow: http://localhost:5000"
curl -s http://localhost:9090 > /dev/null && echo "✓ Prometheus: http://localhost:9090"
curl -s http://localhost:3000 > /dev/null && echo "✓ Grafana: http://localhost:3000 (admin/admin)"

echo ""
echo "All services are running!"
echo ""
echo "View logs: docker-compose logs -f"
echo "Stop all: docker-compose down"