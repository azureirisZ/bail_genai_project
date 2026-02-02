import json
import re
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = "data/processed/segmentation_dataset.jsonl" # Use the segmented data
OUTPUT_FILE = "data/processed/extraction_dataset.jsonl"

# The "Teacher" Rules (Heuristics)
# If these keywords appear, we assume the factor is present.
RULES = {
    "IS_NDPS": [r"ndps", r"narcotic", r"contraband", r"ganja", r"charas", r"heroin"],
    "COMMERCIAL_QUANTITY": [r"commercial quantity", r"huge quantity", r"above commercial"],
    "CHILD_VICTIM": [r"pocso", r"minor victim", r"aged about .* years", r"under section 6"],
    "LONG_CUSTODY": [r"period of custody", r"incarcerated since", r"jail since", r"long incarceration"],
    "BAIL_GRANTED": [r"bail granted", r"application allowed", r"released on bail"]
}

def extract_labels(text):
    labels = {}
    lower_text = text.lower()
    
    for factor, patterns in RULES.items():
        # Check if ANY pattern matches
        if any(re.search(p, lower_text) for p in patterns):
            labels[factor] = 1.0
        else:
            labels[factor] = 0.0
            
    return labels

def main():
    print("🚀 GENERATING EXTRACTION DATA (The 'Detective' Training Set)...")
    
    dataset = []
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                data = json.loads(line)
                # We only want to train on FACTS or ARGUMENTS for extraction
                if data['label'] in ["FACTS", "ARGUMENTS"]:
                    text = data['text']
                    if len(text) < 50: continue # Skip noise
                    
                    labels = extract_labels(text)
                    
                    # Only keep samples that actually have something interesting
                    if sum(labels.values()) > 0:
                        dataset.append({
                            "text": text,
                            "labels": labels
                        })
            except: continue
    
    # Save
    print(f"   📊 Found {len(dataset)} useful training samples.")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for item in dataset:
            f_out.write(json.dumps(item) + "\n")
            
    print(f"✅ Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()