import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec
import re
import numpy as np
import os
import sys

# --- 1. DEFINE ARCHITECTURES (Must match training) ---

class LegalBiLSTM(nn.Module):
    def __init__(self, embedding_matrix):
        super(LegalBiLSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=True)
        self.lstm = nn.LSTM(100, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 4)
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out[:, -1, :])

class BailExtractor(nn.Module):
    def __init__(self, embedding_matrix):
        super(BailExtractor, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=True)
        self.convs = nn.ModuleList([nn.Conv1d(100, 100, k) for k in [3, 4, 5]])
        self.dropout = nn.Dropout(0.5)
        # Heads
        self.head_ndps = nn.Linear(300, 1)
        self.head_qty = nn.Linear(300, 1)
        self.head_child = nn.Linear(300, 1)
        self.head_custody = nn.Linear(300, 1)
        self.head_outcome = nn.Linear(300, 1)

    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)
        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        pooled = [torch.max_pool1d(c, c.shape[2]).squeeze(2) for c in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return {
            "is_ndps": torch.sigmoid(self.head_ndps(cat)),
            "commercial_qty": torch.sigmoid(self.head_qty(cat)),
            "child_victim": torch.sigmoid(self.head_child(cat)),
            "long_custody": torch.sigmoid(self.head_custody(cat)),
            "bail_outcome": torch.sigmoid(self.head_outcome(cat))
        }

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_matrix):
        super(SiameseNetwork, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=True)
        self.lstm = nn.LSTM(100, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 64)
    def forward_one(self, x):
        _, (hidden, _) = self.lstm(self.embedding(x))
        return self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
    def forward(self, x1, x2):
        return F.cosine_similarity(self.forward_one(x1), self.forward_one(x2))

class LegalGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(LegalGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x, hidden=None):
        out, h = self.lstm(self.embedding(x), hidden)
        return self.fc(out), h

# --- 2. HELPER CLASSES ---
class CharTokenizer:
    def __init__(self, filepath):
        # FIX: Read the ACTUAL file to get the exact same 89 characters
        print(f"   📖 Building Tokenizer from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read(1000000)
        
        chars = sorted(list(set(text)))
        self.char2idx = {c: i for i, c in enumerate(chars)}
        self.idx2char = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        print(f"      - Vocab Size: {self.vocab_size} (Should match Model: 89)")

    def encode(self, text):
        return [self.char2idx.get(c, 0) for c in text]
    def decode(self, indices):
        return "".join([self.idx2char.get(i, '') for i in indices])

# --- 3. MAIN SYSTEM ---
def main():
    print("🚀 INITIALIZING BAIL RECKONER SYSTEM...")
    
    # A. Load Embeddings
    print("   1️⃣  Loading Word Embeddings...")
    w2v = Word2Vec.load("weights/legal_embeddings_v1.model")
    matrix = w2v.wv.vectors
    
    # B. Load Models
    print("   2️⃣  Loading Neural Networks...")
    
    segmenter = LegalBiLSTM(matrix)
    segmenter.load_state_dict(torch.load("weights/segmenter_model.pth"))
    segmenter.eval()
    
    extractor = BailExtractor(matrix)
    extractor.load_state_dict(torch.load("weights/extractor_model.pth"))
    extractor.eval()
    
    siamese = SiameseNetwork(matrix)
    siamese.load_state_dict(torch.load("weights/similarity_model.pth"))
    siamese.eval()
    
    # FIX: Initialize Tokenizer from file
    tokenizer = CharTokenizer("data/processed/generation_dataset.txt")
    
    generator = LegalGenerator(tokenizer.vocab_size, 64, 128)
    generator.load_state_dict(torch.load("weights/generator_model.pth"))
    generator.eval()

    print("✅ SYSTEM READY. RUNNING TEST CASE...\n")
    print("-" * 60)

    # --- 4. THE SAMPLE INPUT ---
    RAW_CASE_TEXT = """
    The applicant is accused of possessing 5 kg of Ganja. 
    He has been in custody for 2 years. 
    Counsel for the state argues that this is a commercial quantity under NDPS Act.
    However, the applicant is a juvenile aged 17 years.
    """
    
    print(f"📄 INPUT TEXT:\n{RAW_CASE_TEXT.strip()}\n")
    
    # STEP 1: SEGMENTATION
    print("🔍 STEP 1: SEGMENTATION (Highlighting)")
    sentences = [s.strip() for s in RAW_CASE_TEXT.split('.') if len(s) > 5]
    facts_text = ""
    
    for sent in sentences:
        idxs = [w2v.wv.key_to_index[w] for w in sent.lower().split() if w in w2v.wv]
        if not idxs: idxs = [0]
        tensor_in = torch.tensor([idxs], dtype=torch.long)
        
        with torch.no_grad():
            logits = segmenter(tensor_in)
            pred = torch.argmax(logits).item()
            label = ["FACTS", "ARGUMENTS", "REASONING", "ORDER"][pred]
            
        print(f"   - '{sent}' -> 🏷️  {label}")
        if label == "FACTS": facts_text += sent + " "

    # STEP 2: EXTRACTION
    print("\n🕵️  STEP 2: EXTRACTION (The Detective)")
    if not facts_text: facts_text = RAW_CASE_TEXT
    
    idxs = [w2v.wv.key_to_index[w] for w in facts_text.lower().split() if w in w2v.wv]
    if len(idxs) < 100: idxs += [0] * (100 - len(idxs))
    tensor_in = torch.tensor([idxs[:100]], dtype=torch.long)
    
    with torch.no_grad():
        out = extractor(tensor_in)
    
    print(f"   - Is NDPS (Drugs)?      {out['is_ndps'].item():.4f}")
    print(f"   - Commercial Quantity?  {out['commercial_qty'].item():.4f}")
    print(f"   - Child Victim/Accused? {out['child_victim'].item():.4f}")

    # STEP 3: SIMILARITY
    print("\n⚖️  STEP 3: PRECEDENT SEARCH")
    PAST_CASE = "Applicant found with 20 kg ganja, bail denied due to commercial qty."
    
    idxs1 = [w2v.wv.key_to_index[w] for w in facts_text.lower().split() if w in w2v.wv]
    idxs2 = [w2v.wv.key_to_index[w] for w in PAST_CASE.lower().split() if w in w2v.wv]
    max_len = 100
    idxs1 = (idxs1 + [0]*max_len)[:max_len]
    idxs2 = (idxs2 + [0]*max_len)[:max_len]
    
    score = siamese(torch.tensor([idxs1]), torch.tensor([idxs2])).item()
    print(f"   - Input Case vs: '{PAST_CASE}'")
    print(f"   - Similarity Score: {score*100:.2f}%")

    # STEP 4: GENERATION
    print("\n✍️  STEP 4: DRAFTING ORDER")
    seed = "The court orders"
    print(f"   - Seed: '{seed}'")
    
    input_seq = tokenizer.encode(seed)
    h = None
    generated_text = seed
    
    for _ in range(50): 
        with torch.no_grad():
            x = torch.tensor([input_seq], dtype=torch.long)
            logits, h = generator(x, h)
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=0)
            next_char_idx = torch.multinomial(probs, 1).item()
            next_char = tokenizer.decode([next_char_idx])
            generated_text += next_char
            input_seq.append(next_char_idx)
            input_seq = input_seq[1:]
            
    print(f"   - Result: {generated_text}...")

if __name__ == "__main__":
    main()