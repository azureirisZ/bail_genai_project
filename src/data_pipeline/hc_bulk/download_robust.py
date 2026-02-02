import os
import sys
import time
import random
import logging
import requests
import urllib3
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. System Setup
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
sys.stdout.reconfigure(encoding='utf-8')

# --- CONFIGURATION ---
BASE_URL = "https://indiankanoon.org"
RAW_DIR = "data/raw/hc_bulk"
MAX_WORKERS = 5  # Keep low to avoid aggressive IP bans
PAGES_PER_QUERY = 10  # Scans first 10 pages per topic (approx 100-200 cases each)

# 🚀 EXPANDED DOMAINS & KEYWORDS (Optimized for Hits)
# We removed "doctypes:highcourts" because it causes empty results.
# Instead, we rely on natural language ranking.
SEARCH_QUERIES = [
    # --- DOMAIN 1: CORE BAIL (The Bread & Butter) ---
    ("bail section 439 CrPC granted", "Bail_Regular"),
    ("anticipatory bail section 438 allowed", "Bail_Anticipatory"),
    ("default bail section 167 crpc", "Bail_Default"),
    ("suspension of sentence section 389", "Bail_Suspension"),
    
    # --- DOMAIN 2: SPECIAL ACTS (Hard Logic) ---
    ("bail NDPS commercial quantity", "NDPS_Comm"),
    ("bail NDPS section 37 rigors", "NDPS_Sec37"),
    ("bail POCSO child victim", "POCSO_Child"),
    ("bail PMLA money laundering proceeds", "PMLA_Money"),
    ("bail UAPA terrorist activity", "UAPA_Terror"),
    ("bail SC/ST Act atrocity", "SCST_Act"),
    
    # --- DOMAIN 3: ECONOMIC OFFENCES (White Collar) ---
    ("bail section 420 IPC cheating", "Eco_Cheating"),
    ("bail disproportionate assets corruption", "Eco_Assets"),
    ("bail GST tax evasion", "Eco_Tax"),
    
    # --- DOMAIN 4: VIOLENT CRIMES (Heinous Offences) ---
    ("bail section 302 IPC murder", "Crime_Murder"),
    ("bail section 376 IPC rape", "Crime_Rape"),
    ("bail section 304B dowry death", "Crime_Dowry"),
    ("bail section 307 attempt to murder", "Crime_Attempt"),
    
    # --- DOMAIN 5: ADJACENT LIBERTY DOMAINS (New Expansion) ---
    ("parole granted medical grounds", "Liberty_Parole"),
    ("quashing FIR section 482 CrPC", "Liberty_Quashing"),
    ("habeas corpus illegal detention", "Liberty_Habeas"),
    ("juvenile justice bail section 12", "Liberty_Juvenile")
]

# Random User Agents to mimic real browsers
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/116.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'
]

# Logging Setup
os.makedirs(RAW_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def get_session():
    """Creates a robust session with retry logic."""
    s = requests.Session()
    retries = requests.adapters.Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retries)
    s.mount('https://', adapter)
    return s

def download_single_case(url_data):
    """Worker function to download one case."""
    link, category = url_data
    session = get_session()
    
    try:
        # Extract ID (e.g., /doc/123456/)
        parts = link.strip('/').split('/')
        if 'doc' not in parts: 
            return None
        
        case_id = parts[-1]
        filename = f"{category}_{case_id}.html"
        save_path = os.path.join(RAW_DIR, filename)
        
        # Skip if already exists
        if os.path.exists(save_path): 
            return None

        full_url = urljoin(BASE_URL, link)
        headers = {
            'User-Agent': random.choice(USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Referer': 'https://indiankanoon.org/'
        }
        
        # Random sleep to prevent per-thread blocking
        time.sleep(random.uniform(0.5, 2.0))
        
        r = session.get(full_url, headers=headers, timeout=20, verify=False)
        
        if r.status_code == 200:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(r.text)
            return filename
    except Exception as e:
        return None
    return None

def fetch_page_links(query, page_num):
    """Gets all document links from a search result page."""
    session = get_session()
    url = f"{BASE_URL}/search/"
    params = {'formInput': query, 'pagenum': page_num}
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': 'https://indiankanoon.org/'
    }

    try:
        # Politeness Sleep (Crucial for avoiding 0 results)
        time.sleep(random.uniform(2.0, 4.0)) 
        
        r = session.get(url, params=params, headers=headers, timeout=25, verify=False)
        
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            
            # Check for CAPTCHA/Ban
            page_text = soup.get_text().lower()
            if "robot check" in page_text or "access denied" in page_text:
                logging.error("🚨 CAPTCHA/BLOCK DETECTED! Pausing for 60s...")
                time.sleep(60)
                return "BLOCKED"

            # Extract Links (Universal 'a' tag search)
            all_links = soup.find_all('a', href=True)
            doc_links = set()
            
            for a in all_links:
                href = a['href']
                # Filter for valid document links
                if "/doc/" in href and "fragment" not in href:
                    doc_links.add(href)
            
            # Debug: If 0 links, save page to investigate
            if not doc_links:
                if "No results found" in r.text:
                    logging.warning(f"      ⚠️ No results found for query.")
                else:
                    # Save debug file only if it's weird
                    # debug_file = f"debug_page_{page_num}.html"
                    # with open(debug_file, "w", encoding="utf-8") as f: f.write(r.text)
                    logging.warning(f"      ⚠️ Page loaded but 0 links found.")
                
            return list(doc_links)
            
    except Exception as e:
        logging.error(f"   ⚠️ Network Error on Page {page_num}: {e}")
    return []

def run_robust_ingestion():
    print(f"🚀 Starting ROBUST Ingestion...")
    print(f"📂 Saving to: {RAW_DIR}")
    print(f"⚡ Threads: {MAX_WORKERS}")
    
    total_downloaded = 0

    for query, category in SEARCH_QUERIES:
        logging.info(f"🔎 Domain: {category} | Query: '{query}'")
        
        empty_pages_streak = 0
        
        for page_num in range(0, PAGES_PER_QUERY): 
            logging.info(f"   📄 Scanning Page {page_num}...")
            
            links = fetch_page_links(query, page_num)
            
            if links == "BLOCKED":
                continue # Skip this page, try next after sleep
                
            if not links:
                empty_pages_streak += 1
                if empty_pages_streak >= 3:
                    logging.info(f"      🛑 3 Empty Pages in a row. Skipping rest of {category}.")
                    break
                continue
            
            # Reset streak if we found links
            empty_pages_streak = 0
            
            # Download found links in parallel
            tasks = [(link, category) for link in links]
            new_files_this_page = 0
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(download_single_case, t) for t in tasks]
                for future in as_completed(futures):
                    if future.result():
                        new_files_this_page += 1
                        print(".", end="", flush=True) 
            
            total_downloaded += new_files_this_page
            print(f" [Page {page_num} Done: +{new_files_this_page} files]")

    print(f"\n\n✅ INGESTION COMPLETE.")
    print(f"📦 Total New Files: {total_downloaded}")

if __name__ == "__main__":
    run_robust_ingestion()