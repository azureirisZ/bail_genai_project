import os
import pandas as pd

# We use your OneDrive path since that's where the data lives
PROJECT_ROOT = r"C:\Users\Sastra\OneDrive\Desktop\bail_genai_project"

def main():
    print(f"🔍 Scanning: {PROJECT_ROOT}")
    
    # 1. SEARCH FOR TEXT FILES
    txt_files = []
    print("\n📂 Searching for any .txt file...")
    for root, dirs, files in os.walk(PROJECT_ROOT):
        for f in files:
            if f.endswith(".txt"):
                txt_files.append(os.path.join(root, f))
                # Stop after finding 3 examples to keep output clean
                if len(txt_files) >= 3:
                    break
        if len(txt_files) >= 3:
            break
            
    if len(txt_files) > 0:
        print(f"   ✅ Found text files! Example: {txt_files[0]}")
    else:
        print("   ❌ No .txt files found anywhere.")

    # 2. CHECK THE PARQUET FILE
    matrix_path = os.path.join(PROJECT_ROOT, "data/final/sc_case_factor_matrix.parquet")
    if os.path.exists(matrix_path):
        df = pd.read_parquet(matrix_path)
        print(f"\n📊 Matrix loaded. First Case ID: {df['case_id'].iloc[0]}")
    else:
        print(f"\n❌ Matrix file missing at: {matrix_path}")

if __name__ == "__main__":
    main()
