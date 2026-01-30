import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
import numpy as np
from gensim.models import Word2Vec
import gensim.utils
from tqdm import tqdm

# Ensure Python can find your models
sys.path.append(os.getcwd())
from src.models.generator_attention import LegalAttentionGenerator

# --- CONFIGURATION ---
DATA_FILE = "data/processed/generation_dataset.txt"
EMBEDDING_PATH = "weights/legal_embeddings_v1.model"
SAVE_DIR = "weights/checkpoints"
FINAL_MODEL = "weights/generator_god_mode.pth"

# GOD MODE HYPERPARAMETERS
SEQ_LEN = 30         
BATCH_SIZE = 128     # Higher batch size for speed
EPOCHS = 20          # 20 Epochs on full data is plenty
HIDDEN_DIM = 512     
LAYERS = 2

os.makedirs(SAVE_DIR, exist_ok=True)

class RAMEfficientDataset(Dataset):
    def __init__(self, text_file, word2vec, seq_len):
        self.seq_len = seq_len
        self.wv = word2vec.wv
        print("   📂 Reading Full Corpus...")
        
        # Read the file
        with open(text_file, 'r', encoding='utf-8') as f:
            raw_text = f.read() 
            
        print(f"   Processing text (Removing punctuation & tokenizing)...")
        # FIX: Use gensim's preprocessor. This fixes the "bail." != "bail" bug.
        words = gensim.utils.simple_preprocess(raw_text)
        
        total_words = len(words)
        print(f"   📊 Total Tokens: {total_words} (This should be in the MILLIONS)")
        
        # Create Numpy Array (Saves RAM)
        self.data = np.zeros(total_words, dtype=np.int32)
        valid_count = 0
        
        # Map words to IDs
        for w in tqdm(words):
            if w in self.wv.key_to_index:
                self.data[valid_count] = self.wv.key_to_index[w]
                valid_count += 1
                
        self.data = self.data[:valid_count]
        print(f"   ✅ Final Training Tokens: {len(self.data)}")

    def __len__(self): return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        # Numpy slicing is fast
        x_np = self.data[idx : idx+self.seq_len]
        y_np = self.data[idx+self.seq_len]
        
        return (
            torch.from_numpy(x_np).long(),
            torch.tensor(y_np, dtype=torch.long)
        )

def main():
    print("🚀 STARTING GOD-MODE TRAINING (v2 - Fixed Punctuation)...")
    
    # 1. Load Embeddings
    w2v = Word2Vec.load(EMBEDDING_PATH)
    embed_matrix = w2v.wv.vectors
    
    # 2. Load Data
    dataset = RAMEfficientDataset(DATA_FILE, w2v, SEQ_LEN)
    
    # num_workers=0 is safer on Windows
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 
    
    # 3. Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   ⚙️  Training on: {device}")
    
    model = LegalAttentionGenerator(embed_matrix, hidden_dim=HIDDEN_DIM, num_layers=LAYERS)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for x, y in progress:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(x)
            
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(loader)
        print(f"   ✅ Epoch {epoch+1} Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)
        
        # Save Checkpoint
        if (epoch+1) % 1 == 0:
            torch.save(model.state_dict(), f"{SAVE_DIR}/god_mode_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), FINAL_MODEL)
    print(f"🎉 TRAINING COMPLETE.")

if __name__ == "__main__":
    main()