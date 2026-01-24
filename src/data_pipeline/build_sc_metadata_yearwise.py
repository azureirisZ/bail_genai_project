import pandas as pd
import s3fs
from pathlib import Path
from tqdm import tqdm

fs = s3fs.S3FileSystem(anon=True)

OUTPUT_DIR = Path("data/metadata")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = range(2014, 2025)

def main():
    for year in tqdm(YEARS, desc="Saving year-wise metadata"):
        s3_path = f"indian-supreme-court-judgments/metadata/parquet/year={year}"

        try:
            df = pd.read_parquet(s3_path, filesystem=fs)
        except Exception as e:
            print(f"⚠️ Failed for {year}: {e}")
            continue

        out_path = OUTPUT_DIR / f"sc_metadata_year_{year}.parquet"
        df.to_parquet(out_path, index=False)

        print(f"✅ Saved {len(df)} rows for {year}")

if __name__ == "__main__":
    main()
