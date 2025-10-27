from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from compare_models import ModelComparison
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
import mlflow
from datetime import datetime

app = FastAPI(
    title="RAG Multi-LLM Comparison API",
    description="REST API for querying documents with multiple LLMs",
    version="1.0.0"
)

comparison = None

# Setup MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("rag-api-queries")

# Prometheus metrics
query_counter = Counter('rag_queries_total', 'Total number of queries', ['model'])
query_duration = Histogram('rag_query_duration_seconds', 'Query duration', ['model'])

class QueryRequest(BaseModel):
    question: str
    models: Optional[List[str]] = None
    k: Optional[int] = 3
    parallel: Optional[bool] = True
    track_mlflow: Optional[bool] = True

class QueryResponse(BaseModel):
    question: str
    results: dict
    total_time: float
    mlflow_run_id: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    global comparison
    models = [
        "qwen2.5:7b",
        "codellama:7b-instruct",
        "deepseek-r1:7b",
        "phi3:mini"
    ]
    comparison = ModelComparison(models)
    comparison.setup(vectorstore_path="vectorstore")
    print("RAG system loaded and ready")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

@app.get("/")
async def root():
    return {
        "message": "RAG Multi-LLM Comparison API",
        "version": "1.0.0",
        "features": ["FastAPI", "MLflow Tracking", "Prometheus Metrics", "Parallel Execution"],
        "endpoints": {
            "/query": "POST - Query documents with multiple LLMs",
            "/models": "GET - List available models",
            "/health": "GET - Health check",
            "/metrics": "GET - Prometheus metrics"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "vectorstore": "loaded", "mlflow": "active"}

@app.get("/models")
async def list_models():
    return {"models": comparison.models}

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    models_to_use = request.models if request.models else comparison.models
    
    start = time.time()
    run_id = None
    
    # Start MLflow run if tracking enabled
    if request.track_mlflow:
        mlflow_run = mlflow.start_run(run_name=f"api_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        run_id = mlflow_run.info.run_id
        
        mlflow.log_param("question", request.question)
        mlflow.log_param("num_sources", request.k)
        mlflow.log_param("num_models", len(models_to_use))
        mlflow.log_param("parallel_execution", request.parallel)
        mlflow.log_param("source", "api")
    
    temp_comparison = ModelComparison(models_to_use)
    temp_comparison.rag = comparison.rag
    
    results = temp_comparison.compare(
        question=request.question,
        k=request.k,
        parallel=request.parallel
    )
    
    total_time = time.time() - start
    
    # Log to Prometheus
    for model, result in results.items():
        query_counter.labels(model=model).inc()
        query_duration.labels(model=model).observe(result["time"])
    
    # Log to MLflow
    if request.track_mlflow:
        mlflow.log_metric("total_execution_time", total_time)
        
        for model, result in results.items():
            model_safe = model.replace(":", "_").replace(".", "_")
            mlflow.log_metric(f"{model_safe}_response_time", result["time"])
            mlflow.log_metric(f"{model_safe}_tokens_per_sec", result["metrics"]["tokens_per_second"])
        
        mlflow.end_run()
    
    return QueryResponse(
        question=request.question,
        results=results,
        total_time=round(total_time, 2),
        mlflow_run_id=run_id
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)