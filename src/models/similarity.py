import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_matrix):
        super(SiameseNetwork, self).__init__()
        
        # 1. Shared Embedding Layer (Frozen)
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32), 
            freeze=True
        )
        
        # 2. The "Encoder" (LSTM that summarizes the text)
        self.lstm = nn.LSTM(100, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 64) # Compress to 64-dim vector

    def forward_one(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        # Combine forward and backward states
        context = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(context)

    def forward(self, text_a, text_b):
        # Pass both texts through the SAME network
        vector_a = self.forward_one(text_a)
        vector_b = self.forward_one(text_b)
        
        # Calculate Cosine Similarity
        # Output is between -1 (Opposite) and 1 (Identical)
        return F.cosine_similarity(vector_a, vector_b)