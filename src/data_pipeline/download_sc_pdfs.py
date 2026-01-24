import pandas as pd
import s3fs
from pathlib import Path
from tqdm import tqdm

S3_BUCKET = "indian-supreme-court-judgments"
META_PATH = "data/metadata/sc_criminal_candidate_metadata_2014_2024.parquet"

RAW_DIR = Path("data/raw/sc")
RAW_DIR.mkdir(parents=True, exist_ok=True)

fs = s3fs.S3FileSystem(anon=True)

def main():
    df = pd.read_parquet(META_PATH)

    downloaded = 0
    missing = 0

    print(f"📄 Total metadata rows: {len(df)}")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        if not isinstance(row.get("path"), str):
            missing += 1
            continue

        s3_key = row["path"]

        # Ensure correct prefix
        if not s3_key.startswith("judgments/"):
            s3_key = "judgments/" + s3_key

        s3_path = f"{S3_BUCKET}/{s3_key}"
        local_file = RAW_DIR / Path(s3_key).name

        if local_file.exists():
            continue

        # 🔍 PRE-CHECK existence
        if not fs.exists(s3_path):
            missing += 1
            continue

        try:
            fs.get(s3_path, str(local_file))
            downloaded += 1
        except Exception as e:
            missing += 1

    print("\n✅ DOWNLOAD SUMMARY")
    print(f"Downloaded PDFs : {downloaded}")
    print(f"Missing PDFs    : {missing}")

if __name__ == "__main__":
    main()
