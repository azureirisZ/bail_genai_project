import pandas as pd
import re
from tqdm import tqdm

IN_FILE = r"C:\Users\Sastra\OneDrive\Desktop\bail_genai_project\data/interim/judgment_texts.parquet"
OUT_FILE = r"C:\Users\Sastra\OneDrive\Desktop\bail_genai_project\data/final/case_outcomes.parquet"

GRANTED_PATTERNS = [
    r"bail\s+is\s+granted",
    r"enlarged\s+on\s+bail",
    r"appeal\s+is\s+allowed",
]

REJECTED_PATTERNS = [
    r"bail\s+is\s+rejected",
    r"application\s+is\s+dismissed",
    r"appeal\s+is\s+dismissed",
]

def detect_outcome(text):
    tail = text.lower()[-3000:]
    for p in GRANTED_PATTERNS:
        if re.search(p, tail):
            return "GRANTED"
    for p in REJECTED_PATTERNS:
        if re.search(p, tail):
            return "REJECTED"
    return None

df = pd.read_parquet(IN_FILE)

records = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    outcome = detect_outcome(row["text"])
    if outcome:
        records.append({
            "case_id": row["case_id"],
            "outcome": outcome
        })

out_df = pd.DataFrame(records)
out_df.to_parquet(OUT_FILE)

print(f"✅ Extracted outcomes for {len(out_df)} cases")
