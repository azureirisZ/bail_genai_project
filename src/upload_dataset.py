import os
from huggingface_hub import HfApi

# --- CONFIGURATION ---
# 1. Your Token
TOKEN = "HF_TOKEN_PLACEHOLDER"

# 2. Your Repo
REPO_ID = "anabaena/bail-project-raw-data"

# 3. The CORRECT File Path (Using forward slashes to fix the error)
FILE_PATH = "data/processed/final_training_data.jsonl"
FILE_NAME_ON_HF = "final_training_data.jsonl" 

def main():
    if not os.path.exists(FILE_PATH):
        print(f"❌ Error: Could not find {FILE_PATH}")
        return

    print(f"🚀 Connecting to Hugging Face Repo: {REPO_ID}...")
    api = HfApi()

    try:
        print(f"⬆️  Uploading {FILE_PATH} (This is big, be patient)...")
        api.upload_file(
            path_or_fileobj=FILE_PATH,
            path_in_repo=FILE_NAME_ON_HF,
            repo_id=REPO_ID,
            repo_type="dataset",
            token=TOKEN,
            commit_message="Full 626k case upload"
        )
        print("\n✅ UPLOAD SUCCESSFUL!")
        print(f"🔗 Check it here: https://huggingface.co/datasets/{REPO_ID}")
        
    except Exception as e:
        print(f"\n❌ Upload Failed: {e}")

if __name__ == "__main__":
    main()