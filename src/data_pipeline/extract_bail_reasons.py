import os
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm


SENT_DIR = "data/processed/sc_sentences"
OUT_FILE = "data/final/sc_bail_reason_sentences.parquet"

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

PATTERNS = {
    "BAIL_GRANTED": r"bail (is )?granted|enlarged on bail|released on bail",
    "BAIL_REJECTED": r"bail (is )?rejected|dismissed|not inclined to grant bail",
    "CUSTODY_DURATION": r"custody|incarceration|detained since",
    "SEVERITY": r"heinous|grave offence|serious nature",
    "ANTECEDENTS": r"criminal antecedents|previous cases|past conduct",
    "EVIDENCE": r"prima facie|material on record|evidence",
    "HEALTH": r"medical|health condition|illness",
    "DELAY": r"delay in trial|trial has not commenced",
    "PRECEDENT": r"v\.|vs\.|versus"
}

rows = []

for file in tqdm(list(Path(SENT_DIR).glob("*.txt"))):
    case_id = file.stem
    sentences = file.read_text(encoding="utf-8", errors="ignore").splitlines()

    for s in sentences:
        tags = [k for k, p in PATTERNS.items() if re.search(p, s, re.I)]
        if tags:
            rows.append({
                "case_id": case_id,
                "sentence": s,
                "tags": tags
            })

df = pd.DataFrame(rows)
df.to_parquet(OUT_FILE)

print(f"✅ Saved {len(df)} tagged bail-related sentences")
