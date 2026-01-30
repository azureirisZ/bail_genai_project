import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import sys
from gensim.models import Word2Vec
from tqdm import tqdm

# Import Model
sys.path.append(os.getcwd())
from src.models.extractor import BailExtractor

# --- CONFIGURATION ---
EMBEDDING_PATH = "weights/legal_embeddings_v1.model"
DATA_FILE = "data/processed/extraction_dataset.jsonl"
MODEL_SAVE_PATH = "weights/extractor_model.pth"
BATCH_SIZE = 32
EPOCHS = 5
MAX_LEN = 100

class ExtractionDataset(Dataset):
    def __init__(self, data_file, word2vec):
        self.samples = []
        self.wv = word2vec.wv
        
        print("   📂 Loading Extraction Data...")
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
                # Limit to 30k for speed, remove this line for full training
                if len(self.samples) > 30000: break 
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        text_words = item['text'].lower().split()
        
        # Vectorize
        indices = [self.wv.key_to_index[w] for w in text_words if w in self.wv]
        if not indices: indices = [0]
        
        # Pad/Truncate
        if len(indices) < MAX_LEN: indices += [0] * (MAX_LEN - len(indices))
        else: indices = indices[:MAX_LEN]
        
        labels = item['labels']
        return (
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(labels['IS_NDPS'], dtype=torch.float),
            torch.tensor(labels['COMMERCIAL_QUANTITY'], dtype=torch.float),
            torch.tensor(labels['CHILD_VICTIM'], dtype=torch.float),
            torch.tensor(labels['LONG_CUSTODY'], dtype=torch.float),
            torch.tensor(labels['BAIL_GRANTED'], dtype=torch.float)
        )

def main():
    print("🚀 STARTING MODEL 3 TRAINING (Multi-Task CNN)")
    
    # 1. Load Embeddings
    w2v = Word2Vec.load(EMBEDDING_PATH)
    embedding_matrix = w2v.wv.vectors
    
    # 2. Data
    dataset = ExtractionDataset(DATA_FILE, w2v)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Model
    model = BailExtractor(embedding_matrix)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss() # Binary Cross Entropy
    
    # 4. Train Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            inputs, y_ndps, y_qty, y_child, y_cust, y_out = batch
            
            optimizer.zero_grad()
            preds = model(inputs)
            
            # Calculate Loss for ALL tasks
            loss = (
                criterion(preds['is_ndps'].squeeze(), y_ndps) +
                criterion(preds['commercial_qty'].squeeze(), y_qty) +
                criterion(preds['child_victim'].squeeze(), y_child) +
                criterion(preds['long_custody'].squeeze(), y_cust) +
                criterion(preds['bail_outcome'].squeeze(), y_out)
            )
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"   ✅ Epoch {epoch+1}: Avg Loss {total_loss/len(loader):.4f}")

    # 5. Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"   💾 Saved Model 3 to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()