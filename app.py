import gradio as gr
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")  # Use env var

class RAGInterface:
    def __init__(self):
        self.models = self.get_available_models()
    
    def get_available_models(self):
        try:
            response = requests.get(f"{API_URL}/models")
            return response.json()["models"]
        except:
            return ["qwen2.5:7b", "codellama:7b-instruct", "deepseek-r1:7b", "phi3:mini"]
    
    def query_single(self, question, model, num_sources):
        if not question.strip():
            return "Please enter a question.", ""
        
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={
                    "question": question,
                    "models": [model],
                    "k": num_sources,
                    "parallel": False
                }
            )
            data = response.json()
            result = data["results"][model]
            
            answer = f"**Answer** (took {result['time']}s):\n\n{result['answer']}\n\n"
            answer += f"**Performance:**\n"
            answer += f"- Words: {result['metrics']['word_count']}\n"
            answer += f"- Speed: {result['metrics']['tokens_per_second']} tokens/sec\n\n"
            
            sources = "**Retrieved Sources:**\n\n"
            for i, source in enumerate(result['sources'], 1):
                clean_source = source.replace('\ufeff', '').strip()
                sources += f"{i}. {clean_source}...\n\n"
            
            return answer, sources
        except Exception as e:
            return f"Error: {str(e)}", ""
    
    def query_compare(self, question, models_selected, num_sources):
        if not question.strip():
            return "Please enter a question."
        
        if not models_selected:
            return "Please select at least one model."
        
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={
                    "question": question,
                    "models": models_selected,
                    "k": num_sources,
                    "parallel": True
                }
            )
            data = response.json()
            
            results_text = f"# Question: {question}\n\n"
            results_text += f"**Total Time (Parallel): {data['total_time']}s**\n\n"
            results_text += "=" * 80 + "\n\n"
            
            sorted_results = sorted(
                data["results"].items(), 
                key=lambda x: x[1]["time"]
            )
            
            for model, result in sorted_results:
                results_text += f"## {model}\n"
                results_text += f"**Time:** {result['time']}s | "
                results_text += f"**Speed:** {result['metrics']['tokens_per_second']} tokens/sec | "
                results_text += f"**Words:** {result['metrics']['word_count']}\n\n"
                results_text += f"**Answer:**\n{result['answer']}\n\n"
                results_text += "-" * 80 + "\n\n"
            
            return results_text
        except Exception as e:
            return f"Error: {str(e)}"

def create_interface():
    rag_interface = RAGInterface()
    
    with gr.Blocks(title="RAG Multi-LLM Comparison") as demo:
        gr.Markdown("# RAG Multi-LLM Comparison System")
        gr.Markdown("Query your documents using multiple local LLMs via FastAPI + Ollama")
        
        with gr.Tab("Single Model Query"):
            with gr.Row():
                with gr.Column():
                    question_single = gr.Textbox(
                        label="Question",
                        placeholder="What is machine learning?",
                        lines=3
                    )
                    model_single = gr.Dropdown(
                        choices=rag_interface.models,
                        value=rag_interface.models[0] if rag_interface.models else None,
                        label="Select Model"
                    )
                    num_sources_single = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Number of Sources to Retrieve"
                    )
                    submit_single = gr.Button("Query", variant="primary")
                
                with gr.Column():
                    answer_output = gr.Markdown(label="Answer")
                    sources_output = gr.Markdown(label="Sources")
            
            submit_single.click(
                fn=rag_interface.query_single,
                inputs=[question_single, model_single, num_sources_single],
                outputs=[answer_output, sources_output]
            )
        
        with gr.Tab("Compare Models (Parallel)"):
            with gr.Row():
                with gr.Column():
                    question_compare = gr.Textbox(
                        label="Question",
                        placeholder="Explain RAG in simple terms",
                        lines=3
                    )
                    models_compare = gr.CheckboxGroup(
                        choices=rag_interface.models,
                        value=rag_interface.models[:2] if len(rag_interface.models) >= 2 else rag_interface.models,
                        label="Select Models to Compare"
                    )
                    num_sources_compare = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Number of Sources to Retrieve"
                    )
                    submit_compare = gr.Button("Compare (Parallel)", variant="primary")
                
                with gr.Column():
                    comparison_output = gr.Markdown(label="Comparison Results")
            
            submit_compare.click(
                fn=rag_interface.query_compare,
                inputs=[question_compare, models_compare, num_sources_compare],
                outputs=comparison_output
            )
        
        gr.Markdown("---")
        gr.Markdown("Powered by FastAPI, LangChain, FAISS, and Ollama")
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)