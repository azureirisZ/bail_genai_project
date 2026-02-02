import os
import time
from huggingface_hub import HfApi, CommitOperationAdd

# --- CONFIGURATION ---
USERNAME = "anabaena"  
REPO_NAME = "bail-project-raw-data"
FOLDER_TO_UPLOAD = "data/raw/hc_bulk"
BATCH_SIZE = 50  # Upload 50 files at a time to prevent timeout

api = HfApi()
repo_id = f"{USERNAME}/{REPO_NAME}"

print(f"🚀 Manual Chunked Upload to {repo_id}")
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

# 1. Get list of all files
all_files = []
for root, _, files in os.walk(FOLDER_TO_UPLOAD):
    for file in files:
        full_path = os.path.join(root, file)
        # Calculate path in repo (relative path)
        rel_path = os.path.relpath(full_path, "data/raw")
        all_files.append((full_path, rel_path))

print(f"📦 Found {len(all_files)} files. Uploading in batches of {BATCH_SIZE}...")

# 2. Upload in Batches (Mimics multi_commits manually)
for i in range(0, len(all_files), BATCH_SIZE):
    batch = all_files[i : i + BATCH_SIZE]
    print(f"   📤 Uploading batch {i//BATCH_SIZE + 1} ({len(batch)} files)...")
    
    operations = []
    for full_path, path_in_repo in batch:
        operations.append(
            CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=full_path)
        )
    
    # Retry logic for stability
    for attempt in range(3):
        try:
            api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Upload batch {i}-{i+len(batch)}"
            )
            break # Success, move to next batch
        except Exception as e:
            print(f"      ⚠️ Error (Attempt {attempt+1}): {e}")
            time.sleep(5)

print("✅ All batches uploaded!")