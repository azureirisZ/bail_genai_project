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

# Disable SSL Warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
sys.stdout.reconfigure(encoding='utf-8')

# --- CONFIGURATION ---
BASE_URL = "https://indiankanoon.org"
RAW_DIR = "data/raw/hc_bulk"
MAX_RETRIES = 5

# Queries (Using proper spacing)
SEARCH_QUERIES = [
    "bail IPC doctypes:highcourts",
    "bail NDPS doctypes:highcourts",
    "bail POCSO doctypes:highcourts",
    "bail UAPA doctypes:highcourts"
]

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
]

os.makedirs(RAW_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.FileHandler("ingestion.log", encoding='utf-8'), logging.StreamHandler(sys.stdout)]
)

def fetch_search_results(query, page_num):
    """Fetches search page using proper param encoding."""
    url = f"{BASE_URL}/search/"
    params = {
        'formInput': query,
        'pagenum': page_num
    }
    
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml',
        'Referer': 'https://indiankanoon.org/'
    }

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount('https://', adapter)

    try:
        # Sleep to be polite
        time.sleep(random.uniform(2.0, 4.0))
        
        # Requests handles the ?formInput=... encoding automatically here
        response = session.get(url, params=params, headers=headers, timeout=20, verify=False)
        
        if response.status_code == 200:
            return response.text
        else:
            logging.warning(f"⚠️ Status Code: {response.status_code}")
            return None
            
    except Exception as e:
        logging.error(f"❌ Network Error: {e}")
        return None

def download_case(case_url, category):
    if "/doc/" not in case_url: return 

    try:
        case_id = case_url.strip('/').split('/')[-1]
        safe_id = re.sub(r'[\\/*?:"<>|]', "", case_id)
        filename = f"{category}_{safe_id}.html"
        save_path = os.path.join(RAW_DIR, filename)

        if os.path.exists(save_path): return 

        # Fetch the actual doc
        full_url = urljoin(BASE_URL, case_url)
        # Reuse the logic (simplified for doc fetch)
        headers = {'User-Agent': random.choice(USER_AGENTS)}
        r = requests.get(full_url, headers=headers, timeout=20, verify=False)
        
        if r.status_code == 200:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(r.text)
            logging.info(f"   ✅ Saved: {filename}")
            
    except Exception as e:
        logging.error(f"   ❌ Failed to save {case_url}: {e}")

def run_ingestion():
    logging.info("🚀 Starting Corrected Ingestion...")
    
    for query in SEARCH_QUERIES:
        category = query.split()[1]
        logging.info(f"🔎 Domain: {category}")
        
        for page_num in range(0, 50): 
            logging.info(f"   📄 Scanning Page {page_num}...")
            
            html = fetch_search_results(query, page_num)
            
            if not html:
                logging.warning("      ⚠️ Failed to load page.")
                continue

            soup = BeautifulSoup(html, 'html.parser')
            
            # DEBUG: Print the page title to confirm it's not a Captcha
            page_title = soup.title.string.strip() if soup.title else "No Title"
            logging.info(f"      [Page Title]: {page_title}")

            links = [a['href'] for a in soup.select('.result_title a')]
            valid_links = [l for l in links if "/doc/" in l]
            
            if not valid_links:
                logging.info(f"      ⚠️ No cases found on Page {page_num}. (Check Title above!)")
                # If title says 'Captcha', we know why.
                if "Captcha" in page_title or "Robot" in page_title:
                    logging.error("      🤖 CAPTCHA DETECTED. Stopping script.")
                    return 
                continue
            
            logging.info(f"      Found {len(valid_links)} cases. Downloading...")
            for link in valid_links:
                download_case(link, category)

if __name__ == "__main__":
    run_ingestion()