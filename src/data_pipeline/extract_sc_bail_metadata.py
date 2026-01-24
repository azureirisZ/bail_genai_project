import pandas as pd
import s3fs
import pyarrow.dataset as ds
from tqdm import tqdm

# ---------- CONFIG ----------
START_YEAR = 2014
END_YEAR = 2024

OUTPUT_PATH = "data/metadata/sc_criminal_bail_metadata_2014_2024.parquet"

BAIL_KEYWORDS = [
    "bail",
    "anticipatory bail",
    "regular bail",
    "interim bail",
    "default bail",
    "section 439",
    "section 437",
    "section 438",
    "grant of bail",
    "rejection of bail",
    "cancellation of bail"
]

CRIMINAL_KEYWORDS = [
    "ipc",
    "indian penal code",
    "crpc",
    "code of criminal procedure",
    "section 302",
    "section 376",
    "section 307",
    "section 420",
    "section 406",
    "section 498a",
    "section 34",
    "section 120b",
    "murder",
    "rape",
    "dowry",
    "cheating",
    "criminal conspiracy",
    "forgery",
    "ndps",
    "pocso"
]

# ---------- INIT ----------
import os
os.makedirs("data/metadata", exist_ok=True)

fs = s3fs.S3FileSystem(anon=True)
frames = []

# ---------- LOAD + FILTER ----------
for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="Processing years"):
    try:
        dataset_path = f"indian-supreme-court-judgments/metadata/parquet/year={year}"
        dataset = ds.dataset(dataset_path, filesystem=fs, format="parquet")
        df = dataset.to_table().to_pandas()

        text_blob = (
            df["title"].fillna("") + " " +
            df["description"].fillna("")
        ).str.lower()

        bail_mask = text_blob.apply(
            lambda x: any(k in x for k in BAIL_KEYWORDS)
        )

        criminal_mask = text_blob.apply(
            lambda x: any(k in x for k in CRIMINAL_KEYWORDS)
        )

        final_mask = bail_mask & criminal_mask

        filtered_df = df[final_mask].copy()
        filtered_df["source_year"] = year

        frames.append(filtered_df)

    except Exception as e:
        print(f"⚠️ Failed for year {year}: {e}")

# ---------- SAVE ----------
final_df = pd.concat(frames, ignore_index=True)
final_df.to_parquet(OUTPUT_PATH, index=False)

print(f"✅ Saved {len(final_df)} criminal-law bail cases to:")
print(OUTPUT_PATH)
