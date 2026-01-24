import requests
import pandas as pd
from tqdm import tqdm
import time

SEARCH_URL = "https://dhcsearch.nic.in/dhcsearch/search.do"
OUT_FILE = "data/raw/dhc_html.parquet"

payload_template = {
    "partyName": "",
    "advocateName": "",
    "judgeName": "",
    "fromDate": "01/01/2014",
    "toDate": "31/12/2024",
    "caseType": "",
    "caseNo": "",
    "pageNo": None
}

records = []

for page in tqdm(range(1, 300), desc="Delhi HC search pages"):
    payload = payload_template.copy()
    payload["pageNo"] = page

    try:
        r = requests.post(SEARCH_URL, data=payload, timeout=15)
        if r.status_code != 200:
            break

        data = r.json()
        cases = data.get("resultList", [])
        if not cases:
            break

        for c in cases:
            if "judgmentHtml" not in c:
                continue

            records.append({
                "court": "DHC",
                "case_id": c.get("caseNo"),
                "date": c.get("judgmentDate"),
                "html": c.get("judgmentHtml")
            })

        time.sleep(0.3)

    except Exception:
        break

df = pd.DataFrame(records)
df.to_parquet(OUT_FILE)

print(f"✅ Saved {len(df)} Delhi HC judgments → {OUT_FILE}")
