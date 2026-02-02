import os
from huggingface_hub import HfApi, create_repo

# --- CONFIGURATION ---
TOKEN = "HF_TOKEN_PLACEHOLDER"
REPO_ID = "anabaena/bail-reckoner-models"
WEIGHTS_DIR = "weights/"

def main():
    print(f"🚀 Preparing to upload models to: {REPO_ID}")
    
    # 1. Check if we have all 5 brains
    expected_files = [
        "legal_embeddings_v1.model", # Model 1
        "segmenter_model.pth",       # Model 2
        "extractor_model.pth",       # Model 3
        "similarity_model.pth",      # Model 4
        "generator_model.pth"        # Model 5
    ]
    
    missing = [f for f in expected_files if not os.path.exists(os.path.join(WEIGHTS_DIR, f))]
    if missing:
        print(f"⚠️  Warning: Some models are missing: {missing}")
    else:
        print("✅ All 5 Models detected!")

    # 2. Upload
    api = HfApi()
    try:
        create_repo(repo_id=REPO_ID, token=TOKEN, repo_type="model", exist_ok=True)
        
        print(f"⬆️  Uploading {WEIGHTS_DIR} to Hugging Face...")
        api.upload_folder(
            folder_path=WEIGHTS_DIR,
            repo_id=REPO_ID,
            repo_type="model",
            token=TOKEN,
            commit_message="Upload Final Trained Weights (Models 1-5)"
        )
        print("\n🎉 SUCCESS! All models are live.")
        print(f"🔗 View them here: https://huggingface.co/{REPO_ID}/tree/main")
        
    except Exception as e:
        print(f"\n❌ Upload Failed: {e}")

if __name__ == "__main__":
    main()