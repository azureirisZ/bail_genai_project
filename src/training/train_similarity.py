import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import sys
from gensim.models import Word2Vec
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.models.similarity import SiameseNetwork

# CONFIG
EMBEDDING_PATH = "weights/legal_embeddings_v1.model"
DATA_FILE = "data/processed/similarity_dataset.jsonl"
SAVE_PATH = "weights/similarity_model.pth"
MAX_LEN = 100

class PairDataset(Dataset):
    def __init__(self, data_file, word2vec):
        self.samples = []
        self.wv = word2vec.wv
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self): return len(self.samples)

    def vectorize(self, text):
        indices = [self.wv.key_to_index[w] for w in text.lower().split() if w in self.wv]
        if not indices: indices = [0]
        if len(indices) < MAX_LEN: indices += [0] * (MAX_LEN - len(indices))
        return torch.tensor(indices[:MAX_LEN], dtype=torch.long)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return (
            self.vectorize(item['text_a']),
            self.vectorize(item['text_b']),
            torch.tensor(item['label'], dtype=torch.float)
        )

def main():
    print("🚀 STARTING MODEL 4 TRAINING (Siamese Network)...")
    
    w2v = Word2Vec.load(EMBEDDING_PATH)
    embedding_matrix = w2v.wv.vectors
    
    dataset = PairDataset(DATA_FILE, w2v)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = SiameseNetwork(embedding_matrix)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() # Simple Mean Squared Error for similarity
    
    for epoch in range(3): # 3 Epochs is enough
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            txt_a, txt_b, label = batch
            optimizer.zero_grad()
            
            output = model(txt_a, txt_b)
            loss = criterion(output, label)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"   ✅ Epoch {epoch+1}: Loss {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"   💾 Saved Model 4 to: {SAVE_PATH}")

if __name__ == "__main__":
    main()