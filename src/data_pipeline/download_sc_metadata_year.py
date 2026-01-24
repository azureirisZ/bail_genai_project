import os
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))

META_DIR = os.path.join(PROJECT_ROOT, "data", "metadata", "sc")
os.makedirs(META_DIR, exist_ok=True)

YEAR = 2019  # start with ONE year

S3_URL = (
    f"https://indian-supreme-court-judgments.s3.amazonaws.com/"
    f"metadata/year={YEAR}/"
)

# Known file name pattern
PARQUET_FILE = "judgments_metadata.parquet"

OUT_FILE = os.path.join(META_DIR, f"sc_metadata_{YEAR}.parquet")

def main():
    if os.path.exists(OUT_FILE):
        print("✅ Metadata already exists")
        return

    url = S3_URL + PARQUET_FILE
    print(f"⬇️ Downloading metadata for {YEAR}")

    urllib.request.urlretrieve(url, OUT_FILE)

    print(f"✅ Saved to {OUT_FILE}")

if __name__ == "__main__":
    main()
