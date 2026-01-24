import pandas as pd
import numpy as np
import os
from collections import Counter

# ===== PATHS (WINDOWS SAFE) =====
IN_FILE = r"C:\Users\Sastra\OneDrive\Desktop\bail_genai_project\data\final\sc_bail_reason_sentences_clean.parquet"
OUT_FILE = r"C:\Users\Sastra\OneDrive\Desktop\bail_genai_project\data\final\sc_case_factor_matrix.parquet"

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# ===== LOAD DATA =====
df = pd.read_parquet(IN_FILE)

print(f"DataFrame shape: {df.shape}")
print(f"DataFrame columns: {df.columns.tolist()}")

# Check unique tags in the first few rows to understand the data
print("\nUnique tags found in dataset:")
all_tags = set()
for tags in df["tags"]:
    if isinstance(tags, np.ndarray) or isinstance(tags, list):
        for tag in tags:
            all_tags.add(tag)
    elif pd.notna(tags):
        all_tags.add(tags)

print(f"Total unique tags: {len(all_tags)}")
print("First 20 tags:", sorted(list(all_tags))[:20])

# ===== BUILD CASE → FACTOR COUNTS =====
records = []
case_ids_with_tags = set()

for case_id, group in df.groupby("case_id"):
    tag_counts = Counter()
    
    for tags in group["tags"]:
        # Handle numpy arrays and lists
        if isinstance(tags, np.ndarray) or isinstance(tags, list):
            for tag in tags:
                # Skip if tag is not a string or is empty/NaN
                if isinstance(tag, str) and tag.strip():
                    tag_counts[tag] += 1
        elif isinstance(tags, str) and tags.strip():
            tag_counts[tags] += 1
        # If tags is NaN or None, skip
    
    if tag_counts:  # Only add cases that have valid tags
        case_ids_with_tags.add(case_id)
        row = {"case_id": case_id}
        row.update(tag_counts)
        records.append(row)

print(f"\nProcessed {len(case_ids_with_tags)} cases with tags")

if records:
    # Create DataFrame
    case_factor_df = pd.DataFrame(records)
    
    # Fill NaN values with 0 for all columns except case_id
    # Convert only numeric columns to int
    for col in case_factor_df.columns:
        if col != 'case_id':
            # Fill NaN with 0 and convert to int
            case_factor_df[col] = case_factor_df[col].fillna(0).astype(int)
    
    # Reorder columns: case_id first, then sorted tags
    tag_columns = sorted([col for col in case_factor_df.columns if col != 'case_id'])
    case_factor_df = case_factor_df[['case_id'] + tag_columns]
    
    # ===== SAVE =====
    case_factor_df.to_parquet(OUT_FILE, index=False)
    
    print(f"\n✅ Case-factor matrix saved: {len(case_factor_df)} cases")
    print(f"File saved to: {OUT_FILE}")
    print(f"DataFrame shape: {case_factor_df.shape}")
    
    # Show some statistics
    print(f"\nTotal tags (columns): {len(tag_columns)}")
    print(f"First 10 tag columns: {tag_columns[:10]}")
    
    # Show tag frequency
    print("\nTop 10 most frequent tags across all cases:")
    tag_totals = {}
    for tag in tag_columns:
        tag_totals[tag] = case_factor_df[tag].sum()
    
    for tag, count in sorted(tag_totals.items(), key=lambda x: x[1], reverse=True)[:10]:
        cases_with_tag = (case_factor_df[tag] > 0).sum()
        print(f"  {tag}: {count} total mentions, {cases_with_tag} cases")
    
    # Show a few sample rows
    print(f"\nSample of the matrix (first 3 rows):")
    print(case_factor_df.head(3).to_string())
    
else:
    print("❌ No valid records found.")
