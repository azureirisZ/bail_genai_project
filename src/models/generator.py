import torch
import torch.nn as nn

class LegalGenerator(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=256, num_layers=2):
        super(LegalGenerator, self).__init__()
        
        # 1. Use Pre-trained Embeddings (Don't learn from scratch!)
        # We unfreeze it so it can fine-tune slightly for writing
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32), 
            freeze=False 
        )
        
        embed_dim = embedding_matrix.shape[1]
        vocab_size = embedding_matrix.shape[0]
        
        # 2. Deeper LSTM (2 Layers, 256 Hidden Units)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.3
        )
        
        # 3. Output Layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, (hn, cn) = self.lstm(embedded, hidden)
        logits = self.fc(output)
        return logits, (hn, cn)