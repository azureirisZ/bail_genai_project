import pandas as pd
from pathlib import Path
from tqdm import tqdm

# -----------------------
# CONFIG
# -----------------------

YEARS = range(2014, 2025)
INPUT_DIR = Path("data/metadata")
OUTPUT_PATH = INPUT_DIR / "sc_criminal_candidate_metadata_2014_2024.parquet"

# Extremely broad on purpose
CRIMINAL_HINTS = [
    "criminal",
    "crl",
    "appeal",
    "petition",
    "special leave",
    "slp",
    "state of",
    "vs",
    "versus",
]

# -----------------------
# MAIN
# -----------------------

def load_year(year):
    path = INPUT_DIR / f"sc_metadata_year_{year}.parquet"
    if not path.exists():
        print(f"⚠️ Missing metadata for {year}, skipping")
        return None
    return pd.read_parquet(path)


def is_criminal_candidate(row):
    fields = [
        str(row.get("title", "")),
        str(row.get("case_id", "")),
        str(row.get("court", "")),
        str(row.get("description", "")),
    ]

    blob = " ".join(fields).lower()

    return any(k in blob for k in CRIMINAL_HINTS)


def main():
    collected = []

    for year in tqdm(YEARS, desc="Processing years"):
        df = load_year(year)
        if df is None:
            continue

        df["year"] = year

        df["is_criminal_candidate"] = df.apply(
            is_criminal_candidate, axis=1
        )

        candidates = df[df["is_criminal_candidate"]].copy()
        collected.append(candidates)

    if not collected:
        print("❌ No data collected")
        return

    final_df = pd.concat(collected, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\n✅ Saved {len(final_df)} criminal-law candidate cases")
    print(f"📁 Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
