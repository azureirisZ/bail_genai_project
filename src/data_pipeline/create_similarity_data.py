import json
import random
from tqdm import tqdm

INPUT_FILE = "data/processed/extraction_dataset.jsonl" # Use the extraction data (it has specific labels)
OUTPUT_FILE = "data/processed/similarity_dataset.jsonl"

def get_crime_type(labels):
    if labels['IS_NDPS'] > 0.5: return "NDPS"
    if labels['CHILD_VICTIM'] > 0.5: return "POCSO"
    return "IPC" # Default

def main():
    print("🚀 GENERATING SIMILARITY PAIRS (For Siamese Network)...")
    
    # 1. Load all cases
    cases = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            cases.append(json.loads(line))
            
    if len(cases) < 10:
        print("❌ Not enough data in extraction_dataset.jsonl to build pairs!")
        return

    pairs = []
    # 2. Create Pairs (Positive and Negative)
    for _ in range(5000): # Generate 5000 pairs
        case_a = random.choice(cases)
        case_b = random.choice(cases)
        
        type_a = get_crime_type(case_a['labels'])
        type_b = get_crime_type(case_b['labels'])
        
        # Logic: If crimes are same, they are 'Similar' (1). Else 'Different' (0)
        label = 1.0 if type_a == type_b else 0.0
        
        pairs.append({
            "text_a": case_a['text'],
            "text_b": case_b['text'],
            "label": label
        })
        
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
            
    print(f"✅ Generated {len(pairs)} pairs in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()