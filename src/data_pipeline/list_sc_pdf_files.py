import s3fs

fs = s3fs.S3FileSystem(anon=True)

BUCKET = "indian-supreme-court-judgments"
PREFIX = "judgments/pdf/year=2019"

paths = fs.ls(f"{BUCKET}/{PREFIX}")

print("Sample PDF paths:\n")
for p in paths[:10]:
    print(p)

print(f"\nTotal PDFs found for 2019: {len(paths)}")
