import os
from huggingface_hub import HfApi, create_repo

# --- CONFIGURATION ---
# Ensure your token is set in the environment or paste it here temporarily
TOKEN = os.getenv("HF_TOKEN") 

DATA_REPO_ID = "anabaena/bail-reckoner-data"
FOLDER_TO_UPLOAD = "data/processed"

def main():
    print(f"🚀 UPLOADING DATASETS TO {DATA_REPO_ID}...")
    api = HfApi()
    
    # 1. Create the Repo (if it doesn't exist)
    try:
        create_repo(repo_id=DATA_REPO_ID, repo_type="dataset", token=TOKEN, exist_ok=True)
        print("✅ Repo confirmed.")
    except Exception as e:
        print(f"⚠️  Repo check: {e}")

    # 2. Upload the Heavy Files
    print("⏳ Uploading 1GB+ files. Do not close this window...")
    
    api.upload_folder(
        folder_path=FOLDER_TO_UPLOAD,
        repo_id=DATA_REPO_ID,
        repo_type="dataset",
        token=TOKEN,
        # We specifically want the big text files
        allow_patterns=["*.jsonl", "*.txt"], 
        commit_message="Full training data upload"
    )
    
    print("🎉 UPLOAD COMPLETE. Your data is safe in the cloud.")

if __name__ == "__main__":
    main()