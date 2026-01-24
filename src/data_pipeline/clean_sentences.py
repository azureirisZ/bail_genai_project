import pandas as pd
from tqdm import tqdm

IN_FILE = "data/final/sc_bail_reason_sentences.parquet"
OUT_FILE = "data/final/sc_bail_reason_sentences_clean.parquet"

JUNK_PHRASES = [
    "information made available here is not meant",
    "supreme court of india",
    "copyright",
    "disclaimer",
    "site is designed",
    "all rights reserved"
]

def is_junk(sentence: str) -> bool:
    s = sentence.lower()
    return any(j in s for j in JUNK_PHRASES) or len(s.split()) < 6

def main():
    df = pd.read_parquet(IN_FILE)

    keep_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if not is_junk(row["sentence"]):
            keep_rows.append(row)

    clean_df = pd.DataFrame(keep_rows)
    clean_df.to_parquet(OUT_FILE, index=False)

    print(f"✅ Clean sentences saved: {len(clean_df)}")

if __name__ == "__main__":
    main()
