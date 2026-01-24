import fsspec

fs = fsspec.filesystem("hf")

ROOT = "hf://datasets/ExplodingGradients/indian-high-court-judgments"

print("📁 Root contents:")
print(fs.ls(ROOT))

print("\n📁 metadata/:")
print(fs.ls(f"{ROOT}/metadata"))

print("\n📁 metadata/parquet/:")
print(fs.ls(f"{ROOT}/metadata/parquet"))
