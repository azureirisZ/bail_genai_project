import time
import os
import json
from huggingface_hub import HfApi, create_repo
from datetime import datetime

# --- CONFIGURATION ---
TOKEN = "HF_TOKEN_PLACEHOLDER"
REPO_ID = "anabaena/bail-project-raw-data"

# Files to Merge
SOURCES = [
    "data/processed/aws_sc_data.jsonl",
    "data/processed/scraped_hc_data.jsonl",
    "data/processed/aws_hc_bail_data.jsonl",
    "data/processed/kaggle_bail_data.jsonl"
]
FINAL_OUTPUT = "data/processed/final_training_data.jsonl"

def merge_data():
    print(f"\n🔄 [Sync] Merging latest data...")
    total_count = 0
    
    # Use 'w' to overwrite the file cleanly each time
    with open(FINAL_OUTPUT, 'w', encoding='utf-8') as f_out:
        for source in SOURCES:
            if os.path.exists(source):
                try:
                    with open(source, 'r', encoding='utf-8') as f_in:
                        for line in f_in:
                            if line.strip():
                                f_out.write(line)
                                total_count += 1
                except Exception as e:
                    print(f"   ⚠️ Skipping {source}: {e}")
    
    print(f"   ✅ Merged. Total Cases: {total_count}")
    return total_count

def upload_to_hf(total_count):
    api = HfApi()
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    print(f"⬆️  [Sync] Connecting to Hugging Face...")
    
    # 1. Ensure Repo Exists (Fixes 'Repository Not Found')
    try:
        create_repo(repo_id=REPO_ID, token=TOKEN, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"   ⚠️ Repo check warning (ignoring): {e}")

    # 2. Upload
    print(f"   🚀 Uploading {total_count} cases ({timestamp})...")
    try:
        api.upload_file(
            path_or_fileobj=FINAL_OUTPUT,
            path_in_repo="final_training_data.jsonl",
            repo_id=REPO_ID,
            repo_type="dataset",
            token=TOKEN,
            commit_message=f"Live Sync: {total_count} cases"
        )
        print("   ✅ UPLOAD SUCCESS!")
    except Exception as e:
        print(f"   ❌ Upload Failed: {e}")
        print("      (Check if your Token is 'WRITE' type!)")

def main():
    print("🚀 STARTING LIVE SYNC (Merge + Upload every 5 mins)...")
    
    while True:
        try:
            count = merge_data()
            if count > 0:
                upload_to_hf(count)
            
            print("💤 Sleeping for 5 minutes...")
            time.sleep(300)
            
        except KeyboardInterrupt:
            print("\n🛑 Stopped.")
            break
        except Exception as e:
            print(f"❌ Loop Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()