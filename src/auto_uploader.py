import os
import time
from huggingface_hub import HfApi

# --- CONFIGURATION ---
TOKEN = os.getenv("HF_TOKEN") # Replaced for security  # <--- PASTE YOUR TOKEN
REPO_ID = "anabaena/bail-reckoner-models"
WATCH_DIRS = ["weights/checkpoints", "weights"] # Watch both folders
CHECK_INTERVAL = 300  # Check every 5 minutes

def main():
    api = HfApi()
    uploaded_files = set()
    
    print(f"👀 WATCHDOG ACTIVE.")
    print(f"   - Repo: {REPO_ID}")
    print(f"   - Watching: {WATCH_DIRS}")
    print("   - I will check for new models every 5 minutes.\n")

    while True:
        found_new = False
        
        for folder in WATCH_DIRS:
            if not os.path.exists(folder): continue
            
            # List all model files (.pth, .model)
            files = [f for f in os.listdir(folder) if f.endswith(('.pth', '.model'))]
            
            for filename in files:
                filepath = os.path.join(folder, filename)
                
                # If we haven't uploaded it yet
                if filename not in uploaded_files:
                    # Check if file is still being written (don't upload partial files)
                    # We wait until the file hasn't been modified for 60 seconds
                    last_mod = os.path.getmtime(filepath)
                    if time.time() - last_mod < 60:
                        continue 
                        
                    print(f"⬆️  New Model Detected: {filename}")
                    try:
                        api.upload_file(
                            path_or_fileobj=filepath,
                            path_in_repo=f"checkpoints/{filename}", # Store cleanly in a subfolder
                            repo_id=REPO_ID,
                            repo_type="model",
                            token=TOKEN,
                            commit_message=f"Auto-upload: {filename}"
                        )
                        print(f"✅ SUCCESSFULLY UPLOADED: {filename}")
                        uploaded_files.add(filename)
                        found_new = True
                    except Exception as e:
                        print(f"⚠️  Upload failed (will retry): {e}")

        if not found_new:
            print(".", end="", flush=True) # Heartbeat dot
            
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()