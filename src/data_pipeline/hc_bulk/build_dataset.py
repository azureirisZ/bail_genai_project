import os
import re
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from bs4 import BeautifulSoup

RAW_ROOT = Path("data/raw/hc_bulk")
OUT_PATH = Path("data/processed/hc_bulk_cases.parquet")

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text_from_html(fp: Path) -> str:
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
    return clean_text(soup.get_text(" "))

def extract_text(fp: Path) -> str:
    if fp.suffix.lower() in [".html", ".htm"]:
        return extract_text_from_html(fp)
    elif fp.suffix.lower() in [".txt"]:
        return clean_text(fp.read_text(encoding="utf-8", errors="ignore"))
    else:
        return ""

def build_dataset():
    records = []

    if not RAW_ROOT.exists():
        raise RuntimeError(
            f"❌ Raw HC bulk directory not found: {RAW_ROOT.resolve()}"
        )

    courts = [d for d in RAW_ROOT.iterdir() if d.is_dir()]

    for court_dir in tqdm(courts, desc="Courts"):
        court = court_dir.name.lower()

        years = [y for y in court_dir.iterdir() if y.is_dir()]
        for year_dir in years:
            year = year_dir.name

            files = list(year_dir.glob("*"))
            if not files:
                continue

            for fp in files:
                try:
                    text = extract_text(fp)
                    if len(text) < 500:
                        continue

                    records.append({
                        "court": court,
                        "year": int(year),
                        "case_id": fp.stem,
                        "text": text
                    })

                except Exception as e:
                    print(f"⚠️ Skipped {fp}: {e}")

    if not records:
        raise RuntimeError("❌ No High Court documents were parsed")

    df = pd.DataFrame(records)
    df.to_parquet(OUT_PATH, index=False)

    print(f"✅ Saved dataset: {OUT_PATH}")
    print(f"📄 Total cases: {len(df)}")
    print(df.groupby('court').size().sort_values(ascending=False))

if __name__ == "__main__":
    build_dataset()
