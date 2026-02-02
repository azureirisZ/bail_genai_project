import torch
import torch.nn as nn

class BailExtractor(nn.Module):
    def __init__(self, embedding_matrix, num_filters=100, filter_sizes=[3, 4, 5]):
        super(BailExtractor, self).__init__()
        
        # 1. Load Pretrained Embeddings (Frozen)
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32), 
            freeze=True
        )
        
        # 2. Convolutional Layers (The "Feature Detectors")
        # We use 3 different filter sizes to capture 3-word, 4-word, and 5-word phrases
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in filter_sizes
        ])
        
        self.dropout = nn.Dropout(0.5)
        
        # 3. Multi-Task Output Heads (One classifier per factor)
        cnn_output_dim = num_filters * len(filter_sizes)
        
        self.head_ndps = nn.Linear(cnn_output_dim, 1)      # Is it Drugs?
        self.head_qty = nn.Linear(cnn_output_dim, 1)       # Commercial Quantity?
        self.head_child = nn.Linear(cnn_output_dim, 1)     # Child Victim?
        self.head_custody = nn.Linear(cnn_output_dim, 1)   # Long Custody?
        self.head_outcome = nn.Linear(cnn_output_dim, 1)   # Bail Granted?

    def forward(self, x):
        # x shape: [Batch, Seq_Len]
        embedded = self.embedding(x).permute(0, 2, 1) # [Batch, Embed, Seq]
        
        # Apply Convolutions + ReLU + MaxPool
        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        pooled = [torch.max_pool1d(c, c.shape[2]).squeeze(2) for c in conved]
        
        cat = self.dropout(torch.cat(pooled, dim=1)) # Concatenate features
        
        # Independent Predictions
        return {
            "is_ndps": torch.sigmoid(self.head_ndps(cat)),
            "commercial_qty": torch.sigmoid(self.head_qty(cat)),
            "child_victim": torch.sigmoid(self.head_child(cat)),
            "long_custody": torch.sigmoid(self.head_custody(cat)),
            "bail_outcome": torch.sigmoid(self.head_outcome(cat))
        }