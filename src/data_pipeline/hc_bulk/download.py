import os
import sys
import time
import random
import logging
import requests
import urllib3
import re 
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# 1. Setup: Fix SSL & Windows Display
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
sys.stdout.reconfigure(encoding='utf-8')

# --- CONFIGURATION ---
BASE_URL = "https://indiankanoon.org"
RAW_DIR = "data/raw/hc_bulk"

SEARCH_QUERIES = [
    ("bail IPC", "IPC"), 
    ("bail NDPS", "NDPS"),
    ("bail POCSO", "POCSO"),
    ("bail UAPA", "UAPA")
]

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
]

os.makedirs(RAW_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

def fetch_search_results(query, page_num):
    url = f"{BASE_URL}/search/"
    params = {'formInput': query, 'pagenum': page_num}
    headers = {'User-Agent': random.choice(USER_AGENTS)}

    try:
        # Politeness Sleep
        time.sleep(random.uniform(2.0, 4.0))
        response = requests.get(url, params=params, headers=headers, timeout=30, verify=False)
        return response.text if response.status_code == 200 else None
    except Exception as e:
        logging.error(f"   ⚠️ Network Hiccup: {e}")
        return None

def download_case(case_url, category):
    try:
        # Extract ID and Validate
        case_id = case_url.strip('/').split('/')[-1]
        if not case_id.isdigit(): return 

        filename = f"{category}_{case_id}.html"
        save_path = os.path.join(RAW_DIR, filename)
        
        if os.path.exists(save_path): return 

        full_url = urljoin(BASE_URL, case_url)
        headers = {'User-Agent': random.choice(USER_AGENTS)}
        
        # Download Document
        r = requests.get(full_url, headers=headers, timeout=20, verify=False)
        
        if r.status_code == 200:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(r.text)
            logging.info(f"   ✅ Saved: {filename}")
    except Exception as e:
        logging.error(f"      ❌ Save Error: {e}")

def run_ingestion():
    logging.info("🚀 Starting Unstoppable Ingestion...")
    
    for query, category in SEARCH_QUERIES:
        logging.info(f"🔎 Domain: {category} | Query: {query}")
        
        # INCREASED RANGE: Scans first 50 pages (approx 500-1000 cases per category)
        for page_num in range(0, 50): 
            try:
                logging.info(f"   📄 Scanning Page {page_num}...")
                html = fetch_search_results(query, page_num)
                
                if not html:
                    logging.warning(f"      ⚠️ Page load failed. Skipping to next.")
                    continue

                soup = BeautifulSoup(html, 'html.parser')
                
                # UNIVERSAL GRABBER: Finds ANY link with '/doc/'
                all_links = soup.find_all('a', href=True)
                doc_links = set()
                
                for a in all_links:
                    href = a['href']
                    if "/doc/" in href and "fragment" not in href:
                         doc_links.add(href)
                
                if not doc_links:
                    logging.warning(f"      ⚠️ No docs found on Page {page_num} (Might be end of results).")
                    continue

                logging.info(f"      Found {len(doc_links)} unique cases. Downloading...")
                
                for link in list(doc_links):
                    download_case(link, category)

            except KeyboardInterrupt:
                logging.info("🛑 User stopped script.")
                sys.exit()
            except Exception as e:
                # GLOBAL CRASH GUARD: Catches weird errors and keeps running
                logging.error(f"   🚨 Unexpected Error on Page {page_num}: {e}")
                logging.info("   🔄 Retrying next page in 10 seconds...")
                time.sleep(10)

if __name__ == "__main__":
    run_ingestion()