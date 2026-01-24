import os
import pandas as pd

# ===== CONFIG =====
PROJECT_ROOT = r"C:\Users\Sastra\OneDrive\Desktop\bail_genai_project"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MATRIX_FILE = os.path.join(DATA_DIR, "final", "sc_case_factor_matrix.parquet")

def main():
    print(f"🔍 Scanning Project Root: {PROJECT_ROOT}\n")

    # 1. CHECK THE CASE IDs WE NEED
    if not os.path.exists(MATRIX_FILE):
        print(f"❌ Matrix file missing at: {MATRIX_FILE}")
        return
    
    df = pd.read_parquet(MATRIX_FILE)
    case_ids = df['case_id'].astype(str).tolist()
    print(f"📊 Matrix contains {len(case_ids)} Case IDs.")
    print(f"   Sample IDs: {case_ids[:5]}")
    
    # 2. FIND ALL TEXT FILES ANYWHERE IN 'DATA'
    print("\n📂 Searching for text files in 'data' folder...")
    found_files = {} # filename -> full path
    
    for root, dirs, files in os.walk(DATA_DIR):
        txt_files = [f for f in files if f.endswith('.txt')]
        if txt_files:
            print(f"   Found {len(txt_files)} .txt files in: {root}")
            # Store them for matching
            for f in txt_files:
                found_files[f] = os.path.join(root, f)

    if not found_files:
        print("❌ No .txt files found anywhere in the data folder!")
        return

    # 3. ATTEMPT MATCHING
    print("\n🔗 Testing Linkage (Case ID <-> File Name)...")
    matches = 0
    sample_match_path = None
    
    for cid in case_ids:
        # Check standard naming patterns
        candidates = [
            f"{cid}.txt",           # Standard: 2019_INSC_123.txt
            f"{cid}.txt",           # Lowercase check
            f"judgment_{cid}.txt",  # Common prefix
        ]
        
        matched = False
        for cand in candidates:
            if cand in found_files:
                matches += 1
                matched = True
                sample_match_path = found_files[cand]
                break
        
        # Try checking if case_id is IN the filename (partial match)
        if not matched:
            for fname in list(found_files.keys())[:100]: # Check first 100 files only for speed
                if cid in fname:
                    print(f"   ⚠️ Potential Partial Match found: ID '{cid}' might be in file '{fname}'")
                    break

    print(f"\n✅ Total Matches Found: {matches} / {len(case_ids)}")

    # 4. CONTENT INSPECTION (If we found a match)
    if sample_match_path:
        print(f"\n📝 Content Check for: {os.path.basename(sample_match_path)}")
        try:
            with open(sample_match_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                print(f"   File Size: {len(content)} chars")
                print(f"   Last 500 chars (where outcome usually is):")
                print("-" * 40)
                print(content[-500:].replace("\n", " "))
                print("-" * 40)
        except Exception as e:
            print(f"   ❌ Could not read file: {e}")
    else:
        print("\n❌ CRITICAL: Zero exact matches. Look at the 'Sample IDs' and 'Found files' above to see the difference.")

if __name__ == "__main__":
    main()
