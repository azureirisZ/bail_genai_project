import s3fs

fs = s3fs.S3FileSystem(anon=True)

BUCKET = "indian-supreme-court-judgments"
PREFIX = "judgments/tar/year=2019"  # start with ONE year

paths = fs.ls(f"{BUCKET}/{PREFIX}")

for p in paths[:10]:
    print(p)

print(f"\nTotal TAR files found: {len(paths)}")
