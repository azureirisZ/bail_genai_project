import os
import sys
import logging
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd

# ---------------- CONFIG ----------------
DATASET_NAME = "IndianHighCourtJudgments/metadata"  # streaming dataset
START_YEAR = 2000
OUT_DIR = "data/raw/hc_metadata_by_year"
LOG_DIR = "logs"
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "hc_metadata_2000.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

print("📥 Loading High Court metadata (2000 onwards, streaming mode)...")
logging.info("Script started")

# ---- helper: check if year already done ----
def year_done(year):
    return os.path.exists(os.path.join(OUT_DIR, f"hc_metadata_{year}.parquet"))

# ---- load dataset (STREAMING = SAFE) ----
dataset = load_dataset(
    "datasets/indian-high-court-judgments",
    "metadata",
    split="train",
    streaming=True,
)

current_year = None
buffer = []

try:
    for row in tqdm(dataset, desc="Processing cases"):
        year = int(row.get("decision_year", 0))

        if year < START_YEAR:
            continue

        # if year already processed → skip
        if year_done(year):
            continue

        # year boundary crossed → flush previous year
        if current_year is not None and year != current_year:
            df = pd.DataFrame(buffer)
            out_path = os.path.join(OUT_DIR, f"hc_metadata_{current_year}.parquet")
            df.to_parquet(out_path, index=False)
            logging.info(f"Saved year {current_year} ({len(df)} rows)")
            buffer = []

        current_year = year
        buffer.append(row)

    # ---- save last year ----
    if buffer and current_year is not None:
        df = pd.DataFrame(buffer)
        out_path = os.path.join(OUT_DIR, f"hc_metadata_{current_year}.parquet")
        df.to_parquet(out_path, index=False)
        logging.info(f"Saved year {current_year} ({len(df)} rows)")

except KeyboardInterrupt:
    print("\n🛑 Interrupted safely by user (Ctrl+C)")
    logging.warning("Script interrupted by user")

    if buffer and current_year is not None:
        df = pd.DataFrame(buffer)
        out_path = os.path.join(OUT_DIR, f"hc_metadata_{current_year}.parquet")
        df.to_parquet(out_path, index=False)
        logging.info(f"Partial save for year {current_year} ({len(df)} rows)")

    print(f"💾 Partial data saved for year {current_year}")
    sys.exit(0)

print("✅ Done. All available years saved safely.")
logging.info("Script completed successfully")
