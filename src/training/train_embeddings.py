import os
import json
import multiprocessing
import logging
from gensim.models import Word2Vec
import gensim.utils

# --- CONFIGURATION (Matches your "From Scratch" Doc) ---
DATA_FILE = "data/processed/final_training_data.jsonl"
OUTPUT_DIR = "weights"
MODEL_NAME = "legal_embeddings_v1.model"

# Model Hyperparameters (Optimized for Legal Text)
VECTOR_SIZE = 100    # Size of the "Concept Vector"
WINDOW = 10          # Context window (10 words left/right)
MIN_COUNT = 5        # Ignore rare typos
WORKERS = multiprocessing.cpu_count() # Use all CPU cores
EPOCHS = 5           # Passes over the dataset

# Logging (Shows progress)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class BailCorpus:
    """
    Memory-Safe Iterator: Reads the huge 600k dataset line-by-line
    so your RAM doesn't explode.
    """
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"❌ Missing Data File: {self.filepath}")

        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get('full_text', '')
                    if text:
                        # Preprocessing: Lowercase + Tokenize
                        yield gensim.utils.simple_preprocess(text)
                except json.JSONDecodeError:
                    continue

def main():
    print(f"🚀 STARTING MODEL 1 TRAINING (Legal Embeddings)")
    print(f"   📂 Input Data: {DATA_FILE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Initialize Data Stream
    sentences = BailCorpus(DATA_FILE)
    
    # 2. Define the Skip-Gram Model (sg=1 means Skip-Gram)
    print("\n1️⃣  Building Vocabulary (Scanning all cases)...")
    model = Word2Vec(
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        sg=1  # <--- CRITICAL: This makes it 'Skip-Gram' as per your doc
    )
    
    # 3. Build Vocabulary
    model.build_vocab(sentences)
    print(f"   📖 Vocabulary Size: {len(model.wv)} unique legal terms.")
    
    # 4. Train
    print(f"\n2️⃣  Training for {EPOCHS} Epochs (This is the heavy lifting)...")
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=EPOCHS
    )
    
    # 5. Save Weights
    output_path = os.path.join(OUTPUT_DIR, MODEL_NAME)
    model.save(output_path)
    
    print("\n✅ MODEL 1 COMPLETE.")
    print(f"   💾 Saved to: {output_path}")

    # 6. Sanity Check (Proof it works)
    print("\n🧠 INTELLIGENCE CHECK:")
    try:
        # Check associations
        word = "bail"
        similar = model.wv.most_similar(word, topn=3)
        print(f"   - Concepts close to '{word}': {similar}")
        
        word = "judge"
        similar = model.wv.most_similar(word, topn=3)
        print(f"   - Concepts close to '{word}': {similar}")
    except KeyError:
        print("   (Words not found, maybe dataset was too small?)")

if __name__ == "__main__":
    main()