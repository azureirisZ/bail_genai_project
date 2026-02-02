import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
import json
from gensim.models import Word2Vec
from tqdm import tqdm
import numpy as np

sys.path.append(os.getcwd())
from src.models.generator import LegalGenerator

# --- PRO CONFIGURATION ---
DATA_FILE = "data/processed/generation_dataset.txt"
EMBEDDING_PATH = "weights/legal_embeddings_v1.model"
SAVE_DIR = "weights/checkpoints"
FINAL_MODEL = "weights/generator_model_pro.pth"

# Hyperparameters for "LLM-like" results
SEQ_LEN = 30         # Look back 30 words
BATCH_SIZE = 64      # Process 64 sentences at once
EPOCHS = 50          # Train for a long time
HIDDEN_DIM = 256     # Bigger Brain
LAYERS = 2           # Deeper Brain

# Ensure checkpoint dir exists
os.makedirs(SAVE_DIR, exist_ok=True)

class WordDataset(Dataset):
    def __init__(self, text_file, word2vec, seq_len):
        self.seq_len = seq_len
        self.wv = word2vec.wv
        self.key_to_index = self.wv.key_to_index
        self.index_to_key = self.wv.index_to_key
        
        print("   📂 Tokenizing Text (This converts words to IDs)...")
        # Read text and filter only words that exist in our vocab
        with open(text_file, 'r', encoding='utf-8') as f:
            raw_text = f.read(5000000) # Read 5MB of text (Increase if you have RAM)
            
        words = raw_text.split()
        self.data = [self.key_to_index[w] for w in words if w in self.key_to_index]
        print(f"   📊 Training on {len(self.data)} words.")

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # Input: Sequence of words
        x = torch.tensor(self.data[idx : idx+self.seq_len], dtype=torch.long)
        # Target: The VERY NEXT word
        y = torch.tensor(self.data[idx+1 : idx+self.seq_len+1], dtype=torch.long)
        return x, y

def main():
    print("🚀 STARTING MARATHON TRAINING (Word-Level Generator)...")
    
    # 1. Load Embeddings
    print("   1️⃣  Loading Dictionary...")
    w2v = Word2Vec.load(EMBEDDING_PATH)
    embedding_matrix = w2v.wv.vectors
    
    # 2. Prepare Data
    dataset = WordDataset(DATA_FILE, w2v, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # 3. Initialize Pro Model
    model = LegalGenerator(embedding_matrix, hidden_dim=HIDDEN_DIM, num_layers=LAYERS)
    
    # Use GPU if available (Highly recommended for 50 epochs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   ⚙️  Hardware: {device}")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # 4. The Marathon Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for x, y in progress:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(x)
            
            # Flatten for loss calculation
            loss = criterion(logits.view(-1, len(w2v.wv)), y.view(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Update progress bar
            progress.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(loader)
        print(f"   ✅ Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Adjust Learning Rate
        scheduler.step(avg_loss)
        
        # Save Checkpoint every 5 epochs
        if (epoch+1) % 5 == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"generator_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"   💾 Checkpoint saved: {ckpt_path}")

    # 5. Final Save
    torch.save(model.state_dict(), FINAL_MODEL)
    print(f"\n🎉 MARATHON COMPLETE. Best Model: {FINAL_MODEL}")

if __name__ == "__main__":
    main()