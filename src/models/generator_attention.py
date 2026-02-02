import torch
import torch.nn as nn
import torch.nn.functional as F

class LegalAttentionGenerator(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=512, num_layers=2):
        super(LegalAttentionGenerator, self).__init__()
        
        # 1. Embeddings (Pre-trained Word2Vec)
        # We allow fine-tuning so it adapts to specific drafting styles
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32), 
            freeze=False 
        )
        
        embed_dim = embedding_matrix.shape[1]
        self.vocab_size = embedding_matrix.shape[0]
        self.hidden_dim = hidden_dim
        
        # 2. LSTM Encoder (The "Memory")
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.3
        )
        
        # 3. Attention Layer (The "Focus")
        # Calculates which previous words are relevant to the current prediction
        self.attention = nn.Linear(hidden_dim, 1)
        
        # 4. Output Layer
        self.fc = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, x, hidden=None):
        # x shape: [Batch, Seq_Len]
        embedded = self.embedding(x)
        
        # LSTM Output: [Batch, Seq_Len, Hidden]
        lstm_out, (hn, cn) = self.lstm(embedded, hidden)
        
        # --- ATTENTION MECHANISM ---
        # 1. Calculate "Importance Score" for every word in sequence
        attn_weights = F.softmax(self.attention(lstm_out), dim=1) # [Batch, Seq, 1]
        
        # 2. Create Context Vector (Weighted sum of memory)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1) # [Batch, Hidden]
        
        # 3. Predict next word using the Context
        logits = self.fc(context_vector)
        
        return logits, (hn, cn)