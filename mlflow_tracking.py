import mlflow
import mlflow.pyfunc
import time
from compare_models import ModelComparison
from datetime import datetime
import json

class MLflowRAGTracker:
    def __init__(self, experiment_name="rag-multi-llm-comparison"):
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(experiment_name)
        self.comparison = None
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Experiment: {experiment_name}")
    
    def setup(self, models):
        self.comparison = ModelComparison(models)
        self.comparison.setup(vectorstore_path="vectorstore")
    
    def run_experiment(self, question, k=3, parallel=True, tags=None):
        run_name = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_param("question", question)
            mlflow.log_param("num_sources", k)
            mlflow.log_param("num_models", len(self.comparison.models))
            mlflow.log_param("parallel_execution", parallel)
            mlflow.log_param("timestamp", datetime.now().isoformat())
            
            # Log tags
            if tags:
                mlflow.set_tags(tags)
            
            # Run comparison
            start_time = time.time()
            results = self.comparison.compare(question, k=k, parallel=parallel)
            total_time = time.time() - start_time
            
            # Log overall metrics
            mlflow.log_metric("total_execution_time", total_time)
            
            # Calculate statistics
            response_times = [r["time"] for r in results.values() if r["time"] > 0]
            word_counts = [r["metrics"]["word_count"] for r in results.values()]
            tokens_per_sec = [r["metrics"]["tokens_per_second"] for r in results.values()]
            
            if response_times:
                mlflow.log_metric("avg_response_time", sum(response_times) / len(response_times))
                mlflow.log_metric("min_response_time", min(response_times))
                mlflow.log_metric("max_response_time", max(response_times))
                mlflow.log_metric("avg_word_count", sum(word_counts) / len(word_counts))
                mlflow.log_metric("avg_tokens_per_sec", sum(tokens_per_sec) / len(tokens_per_sec))
            
            # Log per-model metrics and artifacts
            for model, result in results.items():
                model_safe = model.replace(":", "_").replace(".", "_")
                
                # Metrics
                mlflow.log_metric(f"{model_safe}_response_time", result["time"])
                mlflow.log_metric(f"{model_safe}_word_count", result["metrics"]["word_count"])
                mlflow.log_metric(f"{model_safe}_tokens_per_sec", result["metrics"]["tokens_per_second"])
                mlflow.log_metric(f"{model_safe}_response_length", result["metrics"]["response_length"])
                
                # Save answer as artifact
                answer_file = f"{model_safe}_answer.txt"
                with open(answer_file, "w", encoding="utf-8") as f:
                    f.write(f"Model: {model}\n")
                    f.write(f"Question: {question}\n")
                    f.write(f"Response Time: {result['time']}s\n")
                    f.write(f"Tokens/sec: {result['metrics']['tokens_per_second']}\n\n")
                    f.write("Answer:\n")
                    f.write(result["answer"])
                mlflow.log_artifact(answer_file)
            
            # Save complete results as JSON
            results_json = {
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "total_time": total_time,
                "results": results
            }
            with open("experiment_results.json", "w", encoding="utf-8") as f:
                json.dump(results_json, f, indent=2)
            mlflow.log_artifact("experiment_results.json")
            
            run_id = mlflow.active_run().info.run_id
            print(f"\n{'='*80}")
            print(f"Experiment logged to MLflow")
            print(f"Run ID: {run_id}")
            print(f"Run Name: {run_name}")
            print(f"Total Time: {total_time:.2f}s")
            print(f"{'='*80}\n")
            
            return results, run_id
    
    def compare_multiple_questions(self, questions, k=3, parallel=True):
        """Run experiments for multiple questions and compare"""
        all_results = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] Processing: {question}")
            results, run_id = self.run_experiment(
                question, 
                k=k, 
                parallel=parallel,
                tags={"batch": "multi_question", "question_num": str(i)}
            )
            all_results.append({
                "question": question,
                "run_id": run_id,
                "results": results
            })
        
        return all_results

def main():
    print("="*80)
    print("MLflow RAG Experiment Tracking")
    print("="*80)
    
    models = [
        "qwen2.5:7b",
        "codellama:7b-instruct",
        "deepseek-r1:7b",
        "phi3:mini"
    ]
    
    tracker = MLflowRAGTracker()
    tracker.setup(models)
    
    # Single question experiment
    print("\n--- Single Question Experiment ---")
    tracker.run_experiment(
        question="What is machine learning?",
        k=3,
        parallel=True,
        tags={"category": "definition", "domain": "AI"}
    )
    
    # Multiple questions experiment
    print("\n--- Multiple Questions Experiment ---")
    questions = [
        "What is RAG?",
        "Explain vector embeddings",
        "How does machine learning work?"
    ]
    tracker.compare_multiple_questions(questions, k=3, parallel=True)
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print("View results with: mlflow ui")
    print("="*80)

if __name__ == "__main__":
    main()