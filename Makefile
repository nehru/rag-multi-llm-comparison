.PHONY: build up down logs restart clean test

# Build all containers
build:
	docker-compose build

# Start all services
up:
	docker-compose up -d

# Stop all services
down:
	docker-compose down

# View logs
logs:
	docker-compose logs -f

# Restart services
restart:
	docker-compose restart

# Clean everything
clean:
	docker-compose down -v
	docker system prune -f

# Run tests
test:
	docker-compose exec rag-api python test_mlflow_api.py

# Check health
health:
	@echo "Checking service health..."
	@curl -s http://localhost:8000/health || echo "API not ready"
	@curl -s http://localhost:7860 > /dev/null && echo "Gradio UI: Ready" || echo "Gradio UI: Not ready"
	@curl -s http://localhost:5000 > /dev/null && echo "MLflow: Ready" || echo "MLflow: Not ready"
	@curl -s http://localhost:9090 > /dev/null && echo "Prometheus: Ready" || echo "Prometheus: Not ready"
	@curl -s http://localhost:3000 > /dev/null && echo "Grafana: Ready" || echo "Grafana: Not ready"

# Show running containers
ps:
	docker-compose ps

# Execute bash in API container
shell:
	docker-compose exec rag-api bash