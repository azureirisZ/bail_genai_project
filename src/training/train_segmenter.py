import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
from gensim.models import Word2Vec
from tqdm import tqdm

# Import the model we just defined
# (Make sure src/models/__init__.py exists or python path is set)
import sys
sys.path.append(os.getcwd())
from src.models.segmenter import LegalBiLSTM

# --- CONFIGURATION ---
EMBEDDING_PATH = "weights/legal_embeddings_v1.model"
DATA_FILE = "data/processed/segmentation_dataset.jsonl"
MODEL_SAVE_PATH = "weights/segmenter_model.pth"
BATCH_SIZE = 32
EPOCHS = 3
LABELS = {"FACTS": 0, "ARGUMENTS": 1, "REASONING": 2, "ORDER": 3}

class SegmentDataset(Dataset):
    def __init__(self, data_file, word2vec, max_len=50):
        self.samples = []
        self.wv = word2vec.wv
        self.max_len = max_len
        
        print("   📂 Loading Dataset...")
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                self.samples.append(d)
                if len(self.samples) > 20000: break # Train on 20k sentences for speed
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]['text'].lower().split()
        label = LABELS.get(self.samples[idx]['label'], 0)
        
        # Convert words to Indices
        indices = []
        for w in text[:self.max_len]:
            if w in self.wv:
                indices.append(self.wv.key_to_index[w])
            else:
                indices.append(0) # Unknown token
        
        # Padding
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
            
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def main():
    print("🚀 STARTING MODEL 2 TRAINING (BiLSTM Segmenter)")
    
    # 1. Load Embeddings
    print("   🧠 Loading Word2Vec...")
    w2v = Word2Vec.load(EMBEDDING_PATH)
    embedding_matrix = w2v.wv.vectors
    
    # 2. Prepare Data
    dataset = SegmentDataset(DATA_FILE, w2v)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Initialize Model
    model = LegalBiLSTM(embedding_matrix)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Train Loop
    print(f"   ⚔️  Training on {len(dataset)} sentences...")
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        model.train()
        for texts, labels in tqdm(loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(f"   ✅ Epoch {epoch+1}: Loss {total_loss/len(loader):.4f} | Acc: {100 * correct / total:.2f}%")

    # 5. Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"   💾 Saved Model 2 to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()