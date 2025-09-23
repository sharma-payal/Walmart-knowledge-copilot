"""
Walmart Knowledge Copilot - RAG Pipeline (Anti-Repetition Version)
"""

import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import pickle
from typing import List, Dict

class WalmartRAGPipeline:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.embeddings_model = None
        self.generator = None
        self.vector_index = None
        self.documents = []
        self.chunks = []
        
    def load_models(self):
        print("🤖 Loading models...")
        self.embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✅ Embeddings model loaded")
        
        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=150,
            min_length=30,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )
        print("✅ Generation model loaded with anti-repetition")
        
    def load_documents(self):
        print("📚 Loading documents...")
        doc_files = list(self.data_dir.glob("raw/*.txt"))
        
        for file_path in doc_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.documents.append({
                    'filename': file_path.name,
                    'content': content
                })
        print(f"✅ Loaded {len(self.documents)} documents")
        
    def chunk_documents(self):
        print("✂️ Chunking documents...")
        for doc in self.documents:
            content = doc['content']
            sentences = content.split('. ')
            
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) > 50:
                    self.chunks.append({
                        'text': sentence.strip() + '.',
                        'source': doc['filename'],
                        'chunk_id': len(self.chunks)
                    })
        print(f"✅ Created {len(self.chunks)} chunks")
        
    def create_vector_index(self):
        print("🔍 Creating vector index...")
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        embeddings = self.embeddings_model.encode(chunk_texts, show_progress_bar=True)
        
        dimension = embeddings.shape[1]
        self.vector_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.vector_index.add(embeddings.astype('float32'))
        print(f"✅ Vector index created with {self.vector_index.ntotal} vectors")
        
    def retrieve_relevant_chunks(self, query: str, top_k: int = 2) -> List[Dict]:
        query_embedding = self.embeddings_model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.vector_index.search(query_embedding.astype('float32'), top_k)
        
        relevant_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['relevance_score'] = float(score)
                relevant_chunks.append(chunk)
        return relevant_chunks
        
    def generate_answer(self, query: str, relevant_chunks: List[Dict]) -> str:
        context = " ".join([chunk['text'][:200] for chunk in relevant_chunks])
        prompt = f"Answer this Walmart policy question concisely in 2-3 sentences:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        try:
            result = self.generator(prompt, max_length=100, min_length=20, do_sample=True, temperature=0.7)
            answer = result[0]['generated_text'].strip()
            
            # Remove repetition post-processing
            sentences = answer.split('. ')
            unique_sentences = []
            seen = set()
            
            for sentence in sentences:
                clean = sentence.strip().lower()
                if clean not in seen and len(clean) > 10:
                    unique_sentences.append(sentence.strip())
                    seen.add(clean)
            
            final_answer = '. '.join(unique_sentences)
            if not final_answer.endswith('.'):
                final_answer += '.'
                
            return final_answer
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"
        
    def query(self, question: str) -> Dict:
        relevant_chunks = self.retrieve_relevant_chunks(question, top_k=2)
        answer = self.generate_answer(question, relevant_chunks)
        
        return {
            'question': question,
            'answer': answer,
            'sources': [chunk['source'] for chunk in relevant_chunks],
            'relevance_scores': [chunk['relevance_score'] for chunk in relevant_chunks]
        }
        
    def setup_pipeline(self):
        print("🚀 Setting up RAG Pipeline...")
        self.load_models()
        self.load_documents()
        self.chunk_documents()
        self.create_vector_index()
        print("🎉 RAG Pipeline ready!")
        
    def save_index(self, filepath: str = "data/vector_index.faiss"):
        os.makedirs("data", exist_ok=True)
        faiss.write_index(self.vector_index, filepath)
        with open("data/chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"✅ Index saved")
        
    def load_index(self, filepath: str = "data/vector_index.faiss"):
        if os.path.exists(filepath) and os.path.exists("data/chunks.pkl"):
            self.vector_index = faiss.read_index(filepath)
            with open("data/chunks.pkl", "rb") as f:
                self.chunks = pickle.load(f)
            print(f"✅ Index loaded")
            return True
        return False

if __name__ == "__main__":
    rag = WalmartRAGPipeline()
    rag.setup_pipeline()
    result = rag.query("How do I apply for maternity leave?")
    print(f"\n🔎 Query: {result['question']}")
    print(f"📝 Answer: {result['answer']}")
