from huggingface_hub import HfApi, login

# 1. SETUP YOUR TOKEN
# Paste your WRITE token here (starts with hf_...)
TOKEN = "burn_burn" 

# 2. LOGIN
print("🔐 Logging in...")
login(token=TOKEN)

# 3. UPLOAD THE DATABASE
print("☁️ Uploading RAG Database to Hugging Face...")
api = HfApi()

try:
    # Upload FAISS Index
    api.upload_file(
        path_or_fileobj="database/faiss_index.bin",
        path_in_repo="rag_index/faiss_index.bin", # Storing it in a nice subfolder
        repo_id="anabaena/bail-reckoner-data",
        repo_type="dataset"
    )
    print("   ✅ FAISS Index Uploaded!")

    # Upload Case Data
    api.upload_file(
        path_or_fileobj="database/case_data.pkl",
        path_in_repo="rag_index/case_data.pkl",
        repo_id="anabaena/bail-reckoner-data",
        repo_type="dataset"
    )
    print("   ✅ Case Data Uploaded!")
    
    print("\n🎉 SUCCESS: Your Legal Library is now in the cloud.")

except Exception as e:
    print(f"\n❌ ERROR: {e}")