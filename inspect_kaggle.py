import os
import pandas as pd

RAW_DIR = "data/raw/kaggle_hc"

def inspect():
    print(f"🕵️ Inspecting files in {RAW_DIR}...\n")
    
    csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ No CSV files found.")
        return

    for csv_file in csv_files:
        path = os.path.join(RAW_DIR, csv_file)
        try:
            # Read just the first row to get headers
            df = pd.read_csv(path, nrows=2) 
            print(f"📄 File: {csv_file}")
            print(f"   Columns: {list(df.columns)}")
            print("-" * 40)
        except Exception as e:
            print(f"   ⚠️ Could not read {csv_file}: {e}")

if __name__ == "__main__":
    inspect()