import pandas as pd
import fsspec
from tqdm import tqdm

# ---------------- CONFIG ----------------
START_YEAR = 2000
END_YEAR = 2024

HF_DATASET = "hf://datasets/ExplodingGradients/indian-high-court-judgments/metadata/parquet"
OUT_FILE = "data/raw/hc_metadata_2000_onwards.parquet"
# ---------------------------------------

fs = fsspec.filesystem("hf")

all_dfs = []

print("\n📥 Loading High Court metadata (2000 onwards)...\n")

for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="Years"):
    year_path = f"{HF_DATASET}/year={year}"

    try:
        parquet_files = fs.glob(f"{year_path}/*.parquet")

        if not parquet_files:
            continue

        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)
                df["year"] = year
                all_dfs.append(df)
            except Exception as e:
                # silently skip corrupted arrow files
                continue

    except Exception:
        continue

# ---------------- SAFETY ----------------
if not all_dfs:
    raise RuntimeError("❌ No High Court parquet files were loaded")

hc_df = pd.concat(all_dfs, ignore_index=True)

# ---------------- SAVE ------------------
hc_df.to_parquet(OUT_FILE, index=False)

print("\n✅ DONE")
print(f"📦 Saved to: {OUT_FILE}")
print(f"📊 Total rows: {len(hc_df):,}")

if "court" in hc_df.columns:
    print(f"🏛️ High Courts: {hc_df['court'].nunique()}")

print(f"📅 Years covered: {hc_df['year'].min()} – {hc_df['year'].max()}")
