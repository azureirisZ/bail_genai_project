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
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. Setup
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
sys.stdout.reconfigure(encoding='utf-8')

# --- CONFIGURATION ---
BASE_URL = "https://indiankanoon.org"
RAW_DIR = "data/raw/hc_bulk"
MAX_WORKERS = 10

# 🚀 EXPANDED ARSENAL (Deep Dive into Criminal Law)
SEARCH_QUERIES = [
    # 1. The Big Two (Regular & Anticipatory)
    ("section 439 CrPC bail doctypes:highcourts", "Reg_Bail_439"),
    ("section 438 CrPC anticipatory bail doctypes:highcourts", "Ant_Bail_438"),
    
    # 2. Technical Bail (Logic Heavy - Great for training)
    ("section 167(2) default bail doctypes:highcourts", "Default_Bail"),
    ("section 389 suspension of sentence doctypes:highcourts", "Suspension_389"),
    ("section 436 bailable offence doctypes:highcourts", "Bailable_436"),

    # 3. Special Acts (The "Hard" Cases)
    ("section 37 NDPS bail doctypes:highcourts", "NDPS_Sec37"),
    ("section 45 PMLA bail doctypes:highcourts", "PMLA_Sec45"),
    ("section 43D(5) UAPA bail doctypes:highcourts", "UAPA_Sec43D"),
    ("section 12 JJ Act bail juvenile doctypes:highcourts", "Juvenile_Bail"),
    
    # 4. Specific Grounds (Teaching the model "Reasoning")
    ("bail medical grounds doctypes:highcourts", "Ground_Medical"),
    ("bail parity ground doctypes:highcourts", "Ground_Parity"),
    ("bail delay in trial doctypes:highcourts", "Ground_Delay"),
    ("bail false implication doctypes:highcourts", "Ground_FalseImpl")
]

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
]

os.makedirs(RAW_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

def get_session():
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=3, pool_connections=20, pool_maxsize=20)
    s.mount('https://', adapter)
    return s

def download_single_case(url_data):
    """Worker function to download one case."""
    link, category = url_data
    session = get_session()
    
    try:
        case_id = link.strip('/').split('/')[-1]
        if not case_id.isdigit(): return

        filename = f"{category}_{case_id}.html"
        save_path = os.path.join(RAW_DIR, filename)
        
        if os.path.exists(save_path): return 

        full_url = urljoin(BASE_URL, link)
        headers = {'User-Agent': random.choice(USER_AGENTS)}
        
        # Download
        r = session.get(full_url, headers=headers, timeout=15, verify=False)
        
        if r.status_code == 200:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(r.text)
            return filename
    except:
        return None

def fetch_page_links(query, page_num):
    """Gets all links from a search page."""
    session = get_session()
    url = f"{BASE_URL}/search/"
    params = {'formInput': query, 'pagenum': page_num}
    headers = {'User-Agent': random.choice(USER_AGENTS)}

    try:
        # Slight delay to avoid instant ban
        time.sleep(random.uniform(0.5, 1.5)) 
        r = session.get(url, params=params, headers=headers, timeout=20, verify=False)
        
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            all_links = soup.find_all('a', href=True)
            
            doc_links = set()
            for a in all_links:
                if "/doc/" in a['href'] and "fragment" not in a['href']:
                    doc_links.add(a['href'])
            return list(doc_links)
    except Exception as e:
        logging.error(f"   ⚠️ Page {page_num} failed: {e}")
    return []

def run_turbo_ingestion():
    logging.info(f"🚀 Starting TURBO Ingestion ({MAX_WORKERS} Threads)...")
    
    total_downloaded = 0

    for query, category in SEARCH_QUERIES:
        logging.info(f"🔎 Domain: {category} | Query: {query}")
        
        # Scrape 30 pages per category
        for page_num in range(0, 30): 
            logging.info(f"   📄 Scanning Page {page_num}...")
            
            links = fetch_page_links(query, page_num)
            
            if not links:
                logging.info(f"      ⚠️ No links on Page {page_num}. Moving on.")
                continue

            # Prepare data for workers
            tasks = [(link, category) for link in links]
            
            # 🚀 PARALLEL DOWNLOAD
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(download_single_case, t) for t in tasks]
                
                for future in as_completed(futures):
                    res = future.result()
                    if res:
                        total_downloaded += 1
                        # Print a dot for every download to show speed
                        print(".", end="", flush=True) 
            
            print(f" [Page Complete]")

    logging.info(f"\n✅ TURBO COMPLETE. Total New Files: {total_downloaded}")

if __name__ == "__main__":
    run_turbo_ingestion()