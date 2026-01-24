import pandas as pd
import re
import os

# ===== PATHS =====
# Adjust these paths to match your exact folder structure
TEXT_DATA_PATH = r"C:\Users\Sastra\OneDrive\Desktop\bail_genai_project\data\processed" 
MATRIX_FILE = r"C:\Users\Sastra\OneDrive\Desktop\bail_genai_project\data\final\sc_case_factor_matrix.parquet"
OUT_FILE = r"C:\Users\Sastra\OneDrive\Desktop\bail_genai_project\data\final\sc_case_factor_matrix_labeled.parquet"

# ===== PDF PATTERNS [cite: 140, 141] =====
PATTERNS = {
    "GRANTED": [
        r"bail (?:is )?granted",
        r"released on bail",
        r"enlarged on bail",
        r"allow(?:ed)? the appeal",
        r"set aside the (?:impugned )?order"
    ],
    "REJECTED": [
        r"bail (?:is )?rejected",
        r"dismissed",
        r"not inclined to grant",
        r"devoid of merit",
        r"petition is rejected"
    ]
}

def get_outcome(text):
    """
    Scans the last 2000 characters of the judgment (usually the 'Order' section)
    to find the outcome.
    """
    text_lower = text.lower()[-2000:] # Focus on the end of the document
    
    # Check for rejection first (rejections are usually curt/direct)
    for p in PATTERNS["REJECTED"]:
        if re.search(p, text_lower):
            return 0 # REJECTED
            
    # Check for grant
    for p in PATTERNS["GRANTED"]:
        if re.search(p, text_lower):
            return 1 # GRANTED
            
    return -1 # UNKNOWN / AMBIGUOUS

def main():
    # 1. Load your existing matrix
    if not os.path.exists(MATRIX_FILE):
        print(f"❌ Matrix file not found at: {MATRIX_FILE}")
        return
        
    df_matrix = pd.read_parquet(MATRIX_FILE)
    print(f"Loaded matrix with {len(df_matrix)} cases.")

    outcomes = []
    
    # 2. Iterate through cases and find their text files
    print("Extracting outcomes from raw text...")
    for case_id in df_matrix['case_id']:
        # Construct expected filename (assuming case_id matches filename stem)
        # You might need to adjust this depending on how you stored processed text
        txt_path = os.path.join(TEXT_DATA_PATH, f"{case_id}.txt")
        
        outcome = -1
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
                outcome = get_outcome(text)
        
        outcomes.append(outcome)

    # 3. Add column and save
    df_matrix['bail_status'] = outcomes
    
    # Filter out ambiguous cases (-1) for cleaner training data
    df_clean = df_matrix[df_matrix['bail_status'] != -1].copy()
    
    print(f"✅ Outcome extraction complete.")
    print(f"Total Cases: {len(df_matrix)}")
    print(f"Labeled Cases: {len(df_clean)} (Removed {len(df_matrix) - len(df_clean)} ambiguous)")
    print(f"Distribution:\n{df_clean['bail_status'].value_counts()}")
    
    df_clean.to_parquet(OUT_FILE, index=False)
    print(f"💾 Saved labeled matrix to: {OUT_FILE}")

if __name__ == "__main__":
    main()
