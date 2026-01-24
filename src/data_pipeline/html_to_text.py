import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm

METADATA_PATH = "data/metadata/sc_criminal_candidate_metadata_2014_2024.parquet"
OUT_DIR = Path("data/processed/sc_text")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts/styles
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 30]
    return "\n".join(lines)

def main():
    df = pd.read_parquet(METADATA_PATH)

    saved = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        case_id = row["case_id"]
        html = row.get("raw_html")

        if not isinstance(html, str) or len(html) < 500:
            continue

        text = clean_html(html)

        if len(text) < 1000:
            continue

        out_file = OUT_DIR / f"{case_id}.txt"
        out_file.write_text(text, encoding="utf-8")
        saved += 1

    print(f"\n✅ Saved {saved} judgment texts")

if __name__ == "__main__":
    main()
