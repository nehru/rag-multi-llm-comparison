from rag_system import RAGSystem
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

class ModelComparison:
    def __init__(self, models_list):
        self.models = models_list
        self.rag = RAGSystem()
        
    def setup(self, docs_path=None, vectorstore_path="vectorstore"):
        if docs_path:
            self.rag.load_documents(docs_path)
            self.rag.save_vectorstore(vectorstore_path)
        else:
            self.rag.load_vectorstore(vectorstore_path)
    
    def query_single_model(self, model, question, k):
        start_time = time.time()
        try:
            result = self.rag.query(question, model_name=model, k=k)
            elapsed = time.time() - start_time
            
            answer = result["answer"]
            word_count = len(answer.split())
            char_count = len(answer)
            
            return {
                "model": model,
                "answer": answer,
                "time": round(elapsed, 2),
                "sources": result["sources"],
                "metrics": {
                    "response_length": char_count,
                    "word_count": word_count,
                    "tokens_per_second": round(word_count / elapsed, 2) if elapsed > 0 else 0
                }
            }
        except Exception as e:
            return {
                "model": model,
                "answer": f"Error: {str(e)}",
                "time": 0,
                "sources": [],
                "metrics": {"response_length": 0, "word_count": 0, "tokens_per_second": 0}
            }
    
    def compare(self, question, k=3, parallel=True):
        if parallel:
            results = {}
            with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
                future_to_model = {
                    executor.submit(self.query_single_model, model, question, k): model 
                    for model in self.models
                }
                
                for future in as_completed(future_to_model):
                    result = future.result()
                    results[result["model"]] = result
                    print(f"Completed: {result['model']} in {result['time']}s")
            
            return dict(sorted(results.items(), key=lambda x: self.models.index(x[0])))
        else:
            results = {}
            for model in self.models:
                print(f"\nQuerying {model}...")
                results[model] = self.query_single_model(model, question, k)
            return results
    
    def print_comparison(self, results):
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        
        times = [r["time"] for r in results.values() if r["time"] > 0]
        if times:
            print(f"\nPerformance Summary:")
            print(f"  Average Response Time: {statistics.mean(times):.2f}s")
            print(f"  Fastest: {min(times):.2f}s")
            print(f"  Slowest: {max(times):.2f}s")
        
        for model, result in results.items():
            print(f"\nModel: {model}")
            print(f"Time: {result['time']}s")
            print(f"Metrics: {result['metrics']['word_count']} words, "
                  f"{result['metrics']['tokens_per_second']} tokens/sec")
            print(f"Answer: {result['answer'][:300]}...")
            print("-"*80)

def main():
    models = [
        "qwen2.5:7b",
        "codellama:7b-instruct",
        "deepseek-r1:7b",
        "phi3:mini"
    ]
    
    comparison = ModelComparison(models)
    comparison.setup(vectorstore_path="vectorstore")
    
    question = "What is the main concept explained in the documents?"
    
    print("Running parallel comparison...")
    results = comparison.compare(question, k=3, parallel=True)
    comparison.print_comparison(results)

if __name__ == "__main__":
    main()