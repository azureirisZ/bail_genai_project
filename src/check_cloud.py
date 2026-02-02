from huggingface_hub import HfApi

# Your User ID
USER = "anabaena"

def main():
    api = HfApi()
    print(f"🔍 SCANNING CLOUD FOR USER: {USER}...\n")

    # 1. Check Models
    try:
        model_repo = f"{USER}/bail-reckoner-models"
        files = api.list_repo_files(repo_id=model_repo, repo_type="model")
        print(f"✅ FOUND MODEL REPO: {model_repo}")
        for f in files:
            print(f"   - {f}")
    except:
        print(f"❌ Model Repo '{model_repo}' NOT FOUND or Private.")

    print("-" * 30)

    # 2. Check Datasets
    try:
        # We guess the name based on your previous files
        data_repo = f"{USER}/bail-reckoner-data"
        files = api.list_repo_files(repo_id=data_repo, repo_type="dataset")
        print(f"✅ FOUND DATASET REPO: {data_repo}")
        for f in files:
            print(f"   - {f}")
    except:
        print(f"❌ Dataset Repo '{data_repo}' NOT FOUND.")

if __name__ == "__main__":
    main()