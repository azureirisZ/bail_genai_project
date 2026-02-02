import torch
import torch.nn as nn

class LegalBiLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=64, output_dim=4):
        super(LegalBiLSTM, self).__init__()
        
        # 1. Load your trained embeddings (Frozen)
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32), 
            freeze=True # We don't retrain the words, just the logic
        )
        
        # 2. Bi-Directional LSTM (Reads text forwards and backwards)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=True
        )
        
        # 3. Classifier Head (Decides: Facts vs Arguments vs Order)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.embedding(x)      # [Batch, Seq_Len, Embed_Dim]
        lstm_out, _ = self.lstm(embedded) # [Batch, Seq_Len, Hidden*2]
        
        # We only care about the final state of the sentence
        final_state = lstm_out[:, -1, :] 
        
        logits = self.fc(self.dropout(final_state))
        return logits