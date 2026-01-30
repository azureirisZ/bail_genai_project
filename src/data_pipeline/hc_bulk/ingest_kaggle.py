import os
import json
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# --- CONFIGURATION ---
DATASET_ID = "saurabhshahane/high-court-cases-in-india"
RAW_DIR = "data/raw/kaggle_hc"
OUTPUT_FILE = "data/processed/kaggle_bail_data.jsonl"

# We filters for these keywords (Same as your Robust Scraper)
KEYWORDS = [
    "bail", "anticipatory", "parole", "custody", "detention",
    "section 37", "ndps", "uapa", "pmla", "murder", "rape", "302 ipc"
]

def determine_outcome(text):
    """Simple heuristic to label data for training."""
    text = str(text).lower()[-2000:] # Check conclusion
    if any(x in text for x in ["bail granted", "allowed", "released on bail", "application allowed"]):
        return "BAIL_GRANTED"
    if any(x in text for x in ["dismissed", "rejected", "devoid of merit"]):
        return "BAIL_DENIED"
    return "UNCLEAR"

def main():
    print(f"🚀 Connecting to Kaggle API: {DATASET_ID}...")
    
    # 1. Authenticate & Download
    try:
        api = KaggleApi()
        api.authenticate()
        os.makedirs(RAW_DIR, exist_ok=True)
        
        print("   ⬇️  Downloading & Unzipping (This handles ~500MB)...")
        # unzip=True automatically extracts it
        api.dataset_download_files(DATASET_ID, path=RAW_DIR, unzip=True)
        print("   ✅ Download Complete.")
        
    except Exception as e:
        print(f"\n❌ AUTH ERROR: {e}")
        print("👉 Make sure 'kaggle.json' is inside C:\\Users\\<User>\\.kaggle\\")
        return

    # 2. Process the CSVs
    print("⚙️  Processing CSV files...")
    
    csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv')]
    if not csv_files:
        print("❌ No CSV found. Check the download.")
        return

    total_added = 0
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for csv_file in csv_files:
            path = os.path.join(RAW_DIR, csv_file)
            print(f"   📄 Parsing {csv_file}...")
            
            try:
                # Read in chunks to avoid crashing RAM
                for chunk in pd.read_csv(path, chunksize=1000, on_bad_lines='skip'):
                    
                    # Find text column
                    text_col = next((col for col in chunk.columns if 'judgment' in col.lower() or 'text' in col.lower()), None)
                    if not text_col: continue

                    # Filter Rows
                    mask = chunk[text_col].str.contains('|'.join(KEYWORDS), case=False, na=False)
                    relevant_df = chunk[mask]
                    
                    for _, row in relevant_df.iterrows():
                        full_text = row[text_col]
                        if len(str(full_text)) < 500: continue
                        
                        outcome = determine_outcome(full_text)
                        
                        data_point = {
                            "court": "High Court (Kaggle)",
                            "full_text": str(full_text),
                            "outcome": outcome,
                            "source": "Kaggle_API",
                            "instruction": "Analyze this judgment and determine the bail outcome.",
                            "output": f"Outcome: {outcome}."
                        }
                        
                        f_out.write(json.dumps(data_point) + "\n")
                        total_added += 1
                        
            except Exception as e:
                print(f"   ⚠️ Skipping bad file {csv_file}: {e}")

    print(f"\n🎉 KAGGLE INGESTION COMPLETE.")
    print(f"   - Added {total_added} new cases.")
    print(f"   - Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()