import s3fs
from pathlib import Path

fs = s3fs.S3FileSystem(anon=True)

S3_BUCKET = "indian-supreme-court-judgments"
TAR_PREFIX = "judgments/tar"

OUT_DIR = Path("data/raw/sc_tars")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2014, 2025))

def main():
    for year in YEARS:
        tar_name = f"sc_judgments_{year}.tar"
        s3_path = f"{S3_BUCKET}/{TAR_PREFIX}/{tar_name}"
        local_path = OUT_DIR / tar_name

        if local_path.exists():
            print(f"✔️ {tar_name} already exists")
            continue

        print(f"⬇️ Downloading {tar_name}")
        if fs.exists(s3_path):
            fs.get(s3_path, str(local_path))
        else:
            print(f"⚠️ Missing TAR for {year}")

if __name__ == "__main__":
    main()
