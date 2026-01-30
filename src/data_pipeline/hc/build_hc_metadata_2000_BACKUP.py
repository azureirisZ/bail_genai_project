import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_DIR = "data/hc/metadata"
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_YEAR = 2000
END_YEAR = 2024

print("📥 Loading High Court metadata (2000 onwards)...")

all_rows = []

for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="Years"):
    try:
        dataset = load_dataset(
            "indian_high_court_judgments",
            "metadata",
            data_dir=f"metadata/parquet/year={year}",
            split="train"
        )

        df = dataset.to_pandas()
        df["year"] = year
        all_rows.append(df)

    except Exception as e:
        print(f"⚠️ Skipped year {year}: {type(e).__name__}")

if not all_rows:
    raise RuntimeError("❌ No High Court parquet files were loaded")

hc_df = pd.concat(all_rows, ignore_index=True)

out_path = os.path.join(OUTPUT_DIR, "hc_metadata_2000_onwards.parquet")
hc_df.to_parquet(out_path, index=False)

print(f"✅ Saved metadata to {out_path}")
