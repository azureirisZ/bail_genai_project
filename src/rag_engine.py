import os
import faiss
import numpy as np
from gensim.models import Word2Vec
import gensim.utils
import pickle

class LegalRAG:
    def __init__(self, embedding_path="weights/legal_embeddings_v1.model", 
                 index_path="database/faiss_index.bin", 
                 data_path="database/case_data.pkl"):
        
        print("📚 Loading Custom RAG Engine (No Pre-trained Models)...")
        self.index_path = index_path
        self.data_path = data_path
        self.cases = []
        
        # 1. Load YOUR Custom Brain (Word2Vec)
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"❌ Custom Embeddings not found at {embedding_path}")
        
        self.w2v = Word2Vec.load(embedding_path)
        self.vector_size = self.w2v.vector_size
        print(f"✅ Loaded Custom Embeddings (Dim: {self.vector_size})")

    def _get_doc_vector(self, text):
        """Creates a vector for a sentence by averaging its word vectors"""
        words = gensim.utils.simple_preprocess(text)
        word_vectors = []
        
        for w in words:
            if w in self.w2v.wv:
                word_vectors.append(self.w2v.wv[w])
        
        if not word_vectors:
            return np.zeros(self.vector_size) # Return empty vector if no known words
            
        # AVERAGE the vectors (The "Bag of Words" approach)
        return np.mean(word_vectors, axis=0)

    def build_database(self, text_file_path):
        print(f"⚙️ Building Index from {text_file_path}...")
        
        with open(text_file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        # Split into chunks (Pseudo-cases)
        # We split by "SECTION" keywords or just fixed size
        self.cases = [full_text[i:i+1000] for i in range(0, len(full_text), 1000)]
        print(f"   📊 Processed {len(self.cases)} chunks.")

        # Vectorize using YOUR model
        print("   🧮 Vectorizing data (Using Custom Word2Vec)...")
        embeddings = []
        for doc in self.cases:
            embeddings.append(self._get_doc_vector(doc))
            
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS Index
        self.index = faiss.IndexFlatL2(self.vector_size)
        self.index.add(embeddings)
        
        # Save
        os.makedirs("database", exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.data_path, 'wb') as f:
            pickle.dump(self.cases, f)
        print("✅ Database Built & Saved!")

    def load_database(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError("❌ Database not found! Run build_database() first.")
        
        self.index = faiss.read_index(self.index_path)
        with open(self.data_path, 'rb') as f:
            self.cases = pickle.load(f)
        print("✅ Legal Library Loaded.")

    def search(self, query, k=3):
        # 1. Vectorize Query using Custom Brain
        query_vector = self._get_doc_vector(query)
        query_vector = np.array([query_vector]).astype('float32')
        
        # 2. Search FAISS
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.cases):
                results.append(self.cases[idx])
        return results

# --- TEST BLOCK ---
# --- TEST BLOCK ---
if __name__ == "__main__":
    rag = LegalRAG()
    
    # ---------------------------------------------------------
    # STEP 1: BUILD THE DATABASE (Run this ONCE)
    # ---------------------------------------------------------
    print("🚀 STARTING DATABASE BUILD...")
    # Make sure this path points to your text file!
    rag.build_database("data/processed/generation_dataset.txt")
    
    # ---------------------------------------------------------
    # STEP 2: TEST THE SEARCH
    # ---------------------------------------------------------
    print("\n🔍 Testing Search Logic...")
    rag.load_database() # Now this will work because we just built it!
    
    results = rag.search("murder case bail rejected")
    
    for i, res in enumerate(results):
        print(f"\n--- 📄 RESULT {i+1} ---")
        print(res[:300] + "...")