import requests
import time

questions = [
    "What is machine learning?",
    "Explain RAG",
    "What are vector embeddings?",
    "How does AI work?",
    "Define artificial intelligence"
]

for i, question in enumerate(questions, 1):
    print(f"[{i}/{len(questions)}] Querying: {question}")
    
    response = requests.post(
        "http://localhost:8000/query",
        json={
            "question": question,
            "models": ["qwen2.5:7b", "phi3:mini"],
            "k": 3,
            "parallel": True,
            "track_mlflow": True
        }
    )
    
    print(f"  Status: {response.status_code}")
    print(f"  Time: {response.json()['total_time']}s\n")
    
    time.sleep(2)

print("Done! Check Grafana now.")