import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "question": "What is RAG?",
        "models": ["qwen2.5:7b", "phi3:mini"],
        "k": 3,
        "parallel": True,
        "track_mlflow": True
    }
)

print(f"Status: {response.status_code}")
result = response.json()
print(f"MLflow Run ID: {result.get('mlflow_run_id')}")
print(f"Total Time: {result['total_time']}s")
print(f"\nResults:")
for model, data in result['results'].items():
    print(f"  {model}: {data['time']}s - {data['metrics']['tokens_per_second']} tokens/sec")