import os
import re
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = "data/processed/sc_text"
OUTPUT_DIR = "data/processed/sc_sentences"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 30]

def main():
    files = list(Path(INPUT_DIR).glob("*.txt"))
    for f in tqdm(files):
        text = f.read_text(encoding="utf-8", errors="ignore")
        sentences = split_sentences(text)

        out = Path(OUTPUT_DIR) / f.name
        with open(out, "w", encoding="utf-8") as w:
            for s in sentences:
                w.write(s + "\n")

if __name__ == "__main__":
    main()
