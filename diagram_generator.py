import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.lines as mlines

def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(8, 13.5, 'RAG Multi-LLM Comparison System - Production Architecture', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Colors
    color_input = '#E3F2FD'
    color_processing = '#FFF3E0'
    color_storage = '#F3E5F5'
    color_api = '#E8F5E9'
    color_llm = '#FFEBEE'
    color_ui = '#E0F2F1'
    color_monitoring = '#FFF9C4'
    
    # Layer 1: Input & Processing
    input_box = FancyBboxPatch((0.5, 11), 2.5, 1.2, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#1976D2', facecolor=color_input, linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.75, 11.8, 'Documents', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.75, 11.5, 'sample_docs/', fontsize=9, ha='center')
    ax.text(1.75, 11.2, '(.txt files)', fontsize=8, ha='center', style='italic')
    
    # Processing
    processing_box = FancyBboxPatch((4, 11), 2.5, 1.2, 
                                   boxstyle="round,pad=0.1", 
                                   edgecolor='#F57C00', facecolor=color_processing, linewidth=2)
    ax.add_patch(processing_box)
    ax.text(5.25, 11.8, 'Text Processing', fontsize=12, fontweight='bold', ha='center')
    ax.text(5.25, 11.5, 'LangChain', fontsize=9, ha='center')
    ax.text(5.25, 11.2, 'Chunk: 500 chars', fontsize=8, ha='center')
    
    # Embeddings
    embed_box = FancyBboxPatch((7.5, 11), 2.5, 1.2, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='#F57C00', facecolor=color_processing, linewidth=2)
    ax.add_patch(embed_box)
    ax.text(8.75, 11.8, 'Embeddings', fontsize=12, fontweight='bold', ha='center')
    ax.text(8.75, 11.5, 'Ollama', fontsize=9, ha='center')
    ax.text(8.75, 11.2, 'nomic-embed-text', fontsize=8, ha='center', style='italic')
    
    # Layer 2: Vector Store
    vector_box = FancyBboxPatch((4, 8.8), 6, 1.5, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#7B1FA2', facecolor=color_storage, linewidth=2)
    ax.add_patch(vector_box)
    ax.text(7, 9.8, 'FAISS Vector Store', fontsize=13, fontweight='bold', ha='center')
    ax.text(7, 9.4, 'Indexed Document Embeddings', fontsize=9, ha='center')
    ax.text(7, 9.0, 'vectorstore/', fontsize=8, ha='center', style='italic')
    
    # Layer 3: Query Input
    query_box = FancyBboxPatch((0.5, 6.5), 2.5, 1.5, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='#388E3C', facecolor=color_api, linewidth=2)
    ax.add_patch(query_box)
    ax.text(1.75, 7.6, 'User Query', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.75, 7.2, 'Question Input', fontsize=9, ha='center')
    ax.text(1.75, 6.8, 'API / Gradio UI', fontsize=8, ha='center', style='italic')
    
    # Semantic Search
    search_box = FancyBboxPatch((4, 6.5), 6, 1.5, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#388E3C', facecolor=color_api, linewidth=2)
    ax.add_patch(search_box)
    ax.text(7, 7.6, 'Semantic Search', fontsize=13, fontweight='bold', ha='center')
    ax.text(7, 7.2, 'Similarity Search (Top K=3)', fontsize=9, ha='center')
    ax.text(7, 6.8, 'Retrieved Context + Question', fontsize=8, ha='center', style='italic')
    
    # FastAPI Backend
    fastapi_box = FancyBboxPatch((11, 6.5), 2.5, 1.5, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='#388E3C', facecolor=color_api, linewidth=2)
    ax.add_patch(fastapi_box)
    ax.text(12.25, 7.6, 'FastAPI', fontsize=12, fontweight='bold', ha='center')
    ax.text(12.25, 7.2, 'REST API', fontsize=9, ha='center')
    ax.text(12.25, 6.8, 'port 8000', fontsize=8, ha='center', style='italic')
    
    # Layer 4: LLM Models (Parallel Execution)
    llm_models = [
        ('qwen2.5:7b\n4.7 GB', 2, 4.2),
        ('codellama:7b\n3.8 GB', 4.5, 4.2),
        ('deepseek-r1:7b\n4.7 GB', 7, 4.2),
        ('phi3:mini\n2.2 GB', 9.5, 4.2),
    ]
    
    # Parallel execution box
    parallel_box = FancyBboxPatch((1.5, 3.5), 9, 2, 
                                 boxstyle="round,pad=0.1", 
                                 edgecolor='#C62828', facecolor='#FFEBEE', 
                                 linewidth=2, linestyle='--', alpha=0.3)
    ax.add_patch(parallel_box)
    ax.text(6, 5.3, 'Parallel LLM Execution (Ollama)', fontsize=11, 
            fontweight='bold', ha='center', style='italic')
    
    for model, x, y in llm_models:
        llm_box = FancyBboxPatch((x, y), 2, 1, 
                                boxstyle="round,pad=0.05", 
                                edgecolor='#C62828', facecolor=color_llm, linewidth=2)
        ax.add_patch(llm_box)
        ax.text(x+1, y+0.5, model, fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Layer 5: Output & Monitoring
    # Gradio UI
    gradio_box = FancyBboxPatch((0.5, 1), 3, 1.8, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#00897B', facecolor=color_ui, linewidth=2)
    ax.add_patch(gradio_box)
    ax.text(2, 2.4, 'Gradio Web UI', fontsize=12, fontweight='bold', ha='center')
    ax.text(2, 2.0, 'Interactive Interface', fontsize=9, ha='center')
    ax.text(2, 1.7, 'Model Comparison', fontsize=9, ha='center')
    ax.text(2, 1.3, 'port 7860', fontsize=8, ha='center', style='italic')
    
    # MLflow
    mlflow_box = FancyBboxPatch((4.5, 1), 3, 1.8, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#F57C00', facecolor=color_monitoring, linewidth=2)
    ax.add_patch(mlflow_box)
    ax.text(6, 2.4, 'MLflow', fontsize=12, fontweight='bold', ha='center')
    ax.text(6, 2.0, 'Experiment Tracking', fontsize=9, ha='center')
    ax.text(6, 1.7, 'Metrics & Artifacts', fontsize=9, ha='center')
    ax.text(6, 1.3, 'port 5000', fontsize=8, ha='center', style='italic')
    
    # Prometheus
    prom_box = FancyBboxPatch((8.5, 1), 3, 1.8, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='#E65100', facecolor=color_monitoring, linewidth=2)
    ax.add_patch(prom_box)
    ax.text(10, 2.4, 'Prometheus', fontsize=12, fontweight='bold', ha='center')
    ax.text(10, 2.0, 'Metrics Collection', fontsize=9, ha='center')
    ax.text(10, 1.7, 'Time-series DB', fontsize=9, ha='center')
    ax.text(10, 1.3, 'port 9090', fontsize=8, ha='center', style='italic')
    
    # Grafana
    grafana_box = FancyBboxPatch((12.5, 1), 3, 1.8, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='#F57C00', facecolor=color_monitoring, linewidth=2)
    ax.add_patch(grafana_box)
    ax.text(14, 2.4, 'Grafana', fontsize=12, fontweight='bold', ha='center')
    ax.text(14, 2.0, 'Visualization', fontsize=9, ha='center')
    ax.text(14, 1.7, 'Dashboards', fontsize=9, ha='center')
    ax.text(14, 1.3, 'port 3000', fontsize=8, ha='center', style='italic')
    
    # Arrows - Data Flow
    # Input to Processing
    arrow1 = FancyArrowPatch((3, 11.6), (4, 11.6),
                            arrowstyle='->', mutation_scale=25, linewidth=2.5, color='#1976D2')
    ax.add_patch(arrow1)
    
    # Processing to Embeddings
    arrow2 = FancyArrowPatch((6.5, 11.6), (7.5, 11.6),
                            arrowstyle='->', mutation_scale=25, linewidth=2.5, color='#F57C00')
    ax.add_patch(arrow2)
    
    # Embeddings to Vector Store
    arrow3 = FancyArrowPatch((8.75, 11), (7, 10.3),
                            arrowstyle='->', mutation_scale=25, linewidth=2.5, color='#7B1FA2')
    ax.add_patch(arrow3)
    
    # Query to Search
    arrow4 = FancyArrowPatch((3, 7.2), (4, 7.2),
                            arrowstyle='->', mutation_scale=25, linewidth=2.5, color='#388E3C')
    ax.add_patch(arrow4)
    
    # Vector Store to Search
    arrow5 = FancyArrowPatch((7, 8.8), (7, 8),
                            arrowstyle='->', mutation_scale=25, linewidth=2.5, 
                            color='#7B1FA2', linestyle='--')
    ax.add_patch(arrow5)
    ax.text(7.3, 8.4, 'Retrieve', fontsize=9, ha='left',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#7B1FA2'))
    
    # Search to LLMs
    for model, x, y in llm_models:
        arrow = FancyArrowPatch((7, 6.5), (x+1, 5.2),
                               arrowstyle='->', mutation_scale=20, linewidth=2, 
                               color='#388E3C', alpha=0.7)
        ax.add_patch(arrow)
    
    # LLMs to outputs
    arrow6 = FancyArrowPatch((3, 4.2), (2, 2.8),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='#C62828')
    ax.add_patch(arrow6)
    
    arrow7 = FancyArrowPatch((6, 4.2), (6, 2.8),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='#C62828')
    ax.add_patch(arrow7)
    
    arrow8 = FancyArrowPatch((8, 4.2), (10, 2.8),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='#C62828')
    ax.add_patch(arrow8)
    
    arrow9 = FancyArrowPatch((9.5, 4.7), (14, 2.8),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='#C62828')
    ax.add_patch(arrow9)
    
    # FastAPI connections
    arrow10 = FancyArrowPatch((12.25, 6.5), (10, 2.8),
                             arrowstyle='<->', mutation_scale=20, linewidth=2, 
                             color='#388E3C', linestyle='--')
    ax.add_patch(arrow10)
    
    # Prometheus to Grafana
    arrow11 = FancyArrowPatch((11.5, 1.9), (12.5, 1.9),
                             arrowstyle='->', mutation_scale=20, linewidth=2, color='#F57C00')
    ax.add_patch(arrow11)
    
    # Technology Stack Box
    tech_box = FancyBboxPatch((0.3, 0.05), 5, 0.75, 
                             boxstyle="round,pad=0.05", 
                             edgecolor='#424242', facecolor='#FAFAFA', 
                             linewidth=1.5)
    ax.add_patch(tech_box)
    ax.text(2.8, 0.65, 'Technology Stack', fontsize=10, fontweight='bold', ha='center')
    ax.text(0.5, 0.40, 'Framework: LangChain, FastAPI', fontsize=7.5, ha='left')
    ax.text(0.5, 0.20, 'Vector DB: FAISS | LLM: Ollama | UI: Gradio', fontsize=7.5, ha='left')
    
    # Performance Metrics Box
    perf_box = FancyBboxPatch((5.8, 0.05), 4.5, 0.75, 
                             boxstyle="round,pad=0.05", 
                             edgecolor='#424242', facecolor='#E8F5E9', 
                             linewidth=1.5)
    ax.add_patch(perf_box)
    ax.text(8.05, 0.65, 'Performance (RTX 5090)', fontsize=10, fontweight='bold', ha='center')
    ax.text(5.95, 0.40, 'phi3:mini: 2.3s (24 tok/s) | qwen2.5: 4.9s (15 tok/s)', fontsize=7.5, ha='left')
    ax.text(5.95, 0.20, 'Parallel Speedup: 40-60% | Zero API Costs', fontsize=7.5, ha='left')
    
    # Docker Badge
    docker_circle = Circle((11, 0.42), 0.3, color='#2496ED', alpha=0.2)
    ax.add_patch(docker_circle)
    ax.text(11, 0.42, 'Docker', fontsize=9, ha='center', va='center', fontweight='bold', color='#2496ED')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=color_input, edgecolor='#1976D2', label='Input', linewidth=2),
        mpatches.Patch(facecolor=color_processing, edgecolor='#F57C00', label='Processing', linewidth=2),
        mpatches.Patch(facecolor=color_storage, edgecolor='#7B1FA2', label='Storage', linewidth=2),
        mpatches.Patch(facecolor=color_api, edgecolor='#388E3C', label='API/Query', linewidth=2),
        mpatches.Patch(facecolor=color_llm, edgecolor='#C62828', label='LLM Generation', linewidth=2),
        mpatches.Patch(facecolor=color_ui, edgecolor='#00897B', label='User Interface', linewidth=2),
        mpatches.Patch(facecolor=color_monitoring, edgecolor='#F57C00', label='Monitoring', linewidth=2),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8.5, 
             framealpha=0.95, ncol=2, columnspacing=0.5)
    
    plt.tight_layout()
    plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✓ Architecture diagram saved as 'architecture_diagram.png'")
    plt.close()

if __name__ == "__main__":
    create_architecture_diagram()