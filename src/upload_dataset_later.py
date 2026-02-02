import os
from huggingface_hub import HfApi, create_repo

# --- CONFIGURATION ---
# REMEMBER: Use os.getenv or paste the token temporarily when running
TOKEN = os.getenv("HF_TOKEN") 
DATA_REPO_ID = "anabaena/bail-reckoner-data"
FOLDER_TO_UPLOAD = "data/processed"

def main():
    print(f"🚀 PREPARING TO UPLOAD DATASETS TO {DATA_REPO_ID}...")
    api = HfApi()
    
    # 1. Create the Repo (if it doesn't exist)
    try:
        create_repo(repo_id=DATA_REPO_ID, repo_type="dataset", token=TOKEN, exist_ok=True)
        print("✅ Repo created/confirmed.")
    except Exception as e:
        print(f"⚠️  Repo creation check failed: {e}")

    # 2. Upload the Heavy Files
    print("⏳ Uploading 1GB+ files. This will take time...")
    
    # We allow patterns to filter what we upload
    api.upload_folder(
        folder_path=FOLDER_TO_UPLOAD,
        repo_id=DATA_REPO_ID,
        repo_type="dataset",
        token=TOKEN,
        allow_patterns=["*.jsonl", "*.txt"], # Only the big data files
        commit_message="Backup of full training data"
    )
    
    print("🎉 UPLOAD COMPLETE. Your raw data is now safe in the cloud.")

if __name__ == "__main__":
    main()