import os
import time
import requests
import logging
from pathlib import Path
from tqdm import tqdm

# ---------------- CONFIG ---------------- #

BASE_DIR = Path("data/hc_html")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

START_YEAR = 2000
END_YEAR = 2024

REQUEST_DELAY = 1.5  # seconds (polite scraping)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Academic Research Bot)"
}

COURTS = {
    "delhi": {
        "base_url": "https://delhihighcourt.nic.in/judgments",
        "query_param": "year"
    },
    "madras": {
        "base_url": "https://www.mhc.tn.gov.in/judis",
        "query_param": "year"
    },
    "kerala": {
        "base_url": "https://highcourtofkerala.nic.in/judgments",
        "query_param": "year"
    }
}

# ---------------- LOGGING ---------------- #

logging.basicConfig(
    filename=LOG_DIR / "hc_html_scrape.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------- HELPERS ---------------- #

def safe_request(url, params=None):
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=15)
        r.raise_for_status()
        return r.text
    except Exception as e:
        logging.warning(f"Request failed: {url} | {e}")
        return None


def save_html(content, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ---------------- SCRAPER ---------------- #

def scrape_court_year(court_name, court_cfg, year):
    """
    Downloads the judgment listing HTML for a given court + year.
    """
    court_dir = BASE_DIR / court_name / str(year)
    court_dir.mkdir(parents=True, exist_ok=True)

    out_file = court_dir / "index.html"

    # Resume support
    if out_file.exists():
        logging.info(f"Skipping existing: {out_file}")
        return True

    params = {court_cfg["query_param"]: year}
    html = safe_request(court_cfg["base_url"], params=params)

    if html is None:
        logging.error(f"Failed year {year} for {court_name}")
        return False

    save_html(html, out_file)
    time.sleep(REQUEST_DELAY)
    return True


# ---------------- MAIN ---------------- #

def main():
    print("📥 Starting High Court HTML scraping...")
    logging.info("=== HC HTML SCRAPE STARTED ===")

    total_tasks = len(COURTS) * (END_YEAR - START_YEAR + 1)
    completed = 0

    with tqdm(total=total_tasks, desc="Scraping") as pbar:
        for court_name, cfg in COURTS.items():
            for year in range(START_YEAR, END_YEAR + 1):
                try:
                    ok = scrape_court_year(court_name, cfg, year)
                    if not ok:
                        print(f"⚠️ Skipped {court_name} {year}")
                except Exception as e:
                    logging.exception(f"Fatal error {court_name} {year}: {e}")
                completed += 1
                pbar.update(1)

    logging.info("=== HC HTML SCRAPE FINISHED ===")
    print("✅ Scraping complete (or safely skipped).")


if __name__ == "__main__":
    main()
