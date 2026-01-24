import pandas as pd
import os
from collections import Counter

# ===== WINDOWS-SAFE PATHS =====
IN_FILE = r"C:\Users\Sastra\OneDrive\Desktop\bail_genai_project\data\final\sc_bail_reason_sentences_clean.parquet"
OUT_FILE = r"C:\Users\Sastra\OneDrive\Desktop\bail_genai_project\data\final\sc_case_factor_matrix.parquet"

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# ===== LOAD DATA =====
df = pd.read_parquet(IN_FILE)

# Expected columns: case_id | sentence | tags

records = []

for case_id, group in df.groupby("case_id"):
    tag_counts = Counter()

    for tags in group["tags"]:
        for tag in tags:
            tag_counts[tag] += 1

    row = {"case_id": case_id}
    row.update(tag_counts)
    records.append(row)

case_factor_df = pd.DataFrame(records).fillna(0)

# ⚠️ convert ONLY factor columns to int
factor_cols = [c for c in case_factor_df.columns if c != "case_id"]
case_factor_df[factor_cols] = case_factor_df[factor_cols].astype(int)

case_factor_df.to_parquet(OUT_FILE, index=False)

print(f"✅ Case-factor matrix saved: {len(case_factor_df)} cases")
print("Columns:", list(case_factor_df.columns))
