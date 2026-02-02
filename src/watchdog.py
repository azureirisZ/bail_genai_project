import time
import os
from huggingface_hub import HfApi

# --- CONFIGURATION ---
WATCH_FOLDER = "weights/checkpoints"
REPO_ID = "anabaena/bail-reckoner-models"
CHECK_INTERVAL = 300  # Check every 5 minutes

def main():
    print(f"👀 WATCHDOG ACTIVE. Monitoring: {WATCH_FOLDER}")
    print(f"☁️  Target Repo: {REPO_ID}")
    
    api = HfApi()
    uploaded_files = set()
    
    # Pre-fill with files already there so we don't re-upload old ones
    if os.path.exists(WATCH_FOLDER):
        for f in os.listdir(WATCH_FOLDER):
            if f.endswith(".pth"):
                uploaded_files.add(f)
    
    while True:
        try:
            # 1. Scan folder
            if not os.path.exists(WATCH_FOLDER):
                print(f"⏳ Waiting for folder creation...", end="\r")
                time.sleep(60)
                continue

            current_files = [f for f in os.listdir(WATCH_FOLDER) if f.endswith(".pth")]
            
            # 2. Check for NEW files
            for filename in current_files:
                if filename not in uploaded_files:
                    filepath = os.path.join(WATCH_FOLDER, filename)
                    
                    # Wait a bit to ensure file is fully written (safety buffer)
                    file_size = os.path.getsize(filepath)
                    if file_size < 1000000: # Skip if empty/tiny (write in progress)
                        continue
                        
                    print(f"\n🚀 NEW MODEL DETECTED: {filename}")
                    print(f"   ⬆️  Uploading to Hugging Face (Don't close window)...")
                    
                    api.upload_file(
                        path_or_fileobj=filepath,
                        path_in_repo=f"checkpoints/{filename}",
                        repo_id=REPO_ID,
                        repo_type="model"
                    )
                    
                    print(f"   ✅ Upload Complete: {filename}")
                    uploaded_files.add(filename)
                    print("👀 Resume watching...")

            time.sleep(CHECK_INTERVAL)
            
        except Exception as e:
            print(f"⚠️  Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()