import pandas as pd
import s3fs
from tqdm import tqdm
from pathlib import Path

# ---------------- CONFIG ----------------
S3_BASE = "indian-high-court-judgments/metadata/parquet"
YEARS = range(1950, 2024)

TARGET_COURTS = {
    "Delhi High Court": "hc_delhi_html.parquet",
    "Kerala High Court": "hc_kerala_html.parquet",
    "Madras High Court": "hc_madras_html.parquet",
    "Bombay High Court": "hc_bombay_html.parquet",
    "Calcutta High Court": "hc_calcutta_html.parquet",
    "Allahabad High Court": "hc_allahabad_html.parquet",
    "Karnataka High Court": "hc_karnataka_html.parquet",
    "Rajasthan High Court": "hc_rajasthan_html.parquet",
    "Telangana High Court": "hc_telangana_html.parquet",
    "Andhra Pradesh High Court": "hc_andhra_html.parquet",
    "Punjab and Haryana High Court": "hc_punjab_haryana_html.parquet",
}

OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------

fs = s3fs.S3FileSystem(anon=True)
all_rows = []

print("\n📥 Loading High Court metadata by year...\n")

for year in tqdm(YEARS, desc="Years"):
    s3_path = f"s3://{S3_BASE}/year={year}"
    try:
        df = pd.read_parquet(
            s3_path,
            filesystem=fs,
            engine="pyarrow"
        )

        if len(df) == 0:
            continue

        df["year"] = year
        all_rows.append(df)

    except Exception as e:
        print(f"⚠️ Skipped year {year}: {type(e).__name__}")
        continue

# --------- SAFETY CHECK ----------
if not all_rows:
    raise RuntimeError(
        "❌ No High Court data loaded. "
        "Check S3 path or network connectivity."
    )

hc_df = pd.concat(all_rows, ignore_index=True)
print(f"\n✅ Loaded {len(hc_df):,} High Court records")

# ---------- STANDARDIZE ----------
hc_df.columns = [c.lower() for c in hc_df.columns]

RENAMES = {
    "court_name": "court",
    "html_path": "judgment_html_path",
    "judgement_html_path": "judgment_html_path"
}

for k, v in RENAMES.items():
    if k in hc_df.columns:
        hc_df.rename(columns={k: v}, inplace=True)

# ---------- FILTER CRIMINAL / BAIL ----------
KEYWORDS = ["bail", "anticipatory", "criminal", "crl"]

def is_relevant(row):
    text = " ".join([
        str(row.get("case_title", "")),
        str(row.get("case_type", "")),
    ]).lower()
    return any(k in text for k in KEYWORDS)

hc_df = hc_df[hc_df.apply(is_relevant, axis=1)]
print(f"⚖️ After bail/criminal filter: {len(hc_df):,}")

# ---------- SAVE PER COURT ----------
for court, filename in TARGET_COURTS.items():
    subset = hc_df[hc_df["court"] == court]

    if subset.empty:
        print(f"⚠️ No data for {court}")
        continue

    path = OUT_DIR / filename
    subset.to_parquet(path, index=False)
