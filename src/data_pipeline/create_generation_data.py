import json
import os
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = "data/processed/segmentation_dataset.jsonl"
OUTPUT_FILE = "data/processed/generation_dataset.txt"

def main():
    print("🚀 EXTRACTING FULL TEXT CORPUS (God Mode)...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    count = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
            for line in tqdm(f_in):
                try:
                    data = json.loads(line)
                    text = data.get('text', '').strip()
                    
                    # FIX: No filtering. We take EVERYTHING.
                    if len(text) > 5: 
                        f_out.write(text + "\n")
                        count += 1
                except: continue
                
    print(f"✅ Text Corpus Ready: {OUTPUT_FILE}")
    print(f"   📊 Extracted {count} lines of legal text.")

if __name__ == "__main__":
    main()