import requests
import pandas as pd
import time
from bs4 import BeautifulSoup
from tqdm import tqdm

URL = "https://hckerala.gov.in/judgments.php"

records = []
session = requests.Session()

for page in tqdm(range(1, 201), desc="Kerala HC"):
    payload = {
        "page": page,
        "submit": "Search"
    }

    r = session.post(URL, data=payload, timeout=20)
    if r.status_code != 200:
        break

    soup = BeautifulSoup(r.text, "html.parser")
    links = soup.select("a[href*='judgment']")

    if not links:
        break

    for a in links:
        case_url = "https://hckerala.gov.in/" + a["href"]
        html = session.get(case_url, timeout=20).text

        records.append({
            "court": "KHC",
            "url": case_url,
            "html": html
        })

        time.sleep(0.3)

df = pd.DataFrame(records)
df.to_parquet("data/raw/khc_html.parquet")

print(f"✅ Kerala HC saved: {len(df)} cases")
