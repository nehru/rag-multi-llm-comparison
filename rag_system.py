import os
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader

class RAGSystem:
    def __init__(self, embedding_model="nomic-embed-text", chunk_size=500, chunk_overlap=50):
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = None
        
    def load_documents(self, directory_path):
        loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        splits = text_splitter.split_documents(documents)
        
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        print(f"Loaded {len(documents)} documents, created {len(splits)} chunks")
        
    def save_vectorstore(self, path="vectorstore"):
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"Vector store saved to {path}")
            
    def load_vectorstore(self, path="vectorstore"):
        self.vectorstore = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        print(f"Vector store loaded from {path}")
        
    def query(self, question, model_name="qwen2.5:7b", k=3):
        if not self.vectorstore:
            raise ValueError("No vector store loaded. Load documents first.")
        
        docs = self.vectorstore.similarity_search(question, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""Answer the question based on the following context:

Context:
{context}

Question: {question}

Answer:"""
        
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        answer = response['message']['content']
        
        return {
            "answer": answer,
            "sources": [doc.page_content[:200] for doc in docs]
        }