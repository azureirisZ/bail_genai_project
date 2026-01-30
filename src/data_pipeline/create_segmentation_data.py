import json
import re
from tqdm import tqdm

INPUT_FILE = "data/processed/final_training_data.jsonl"
OUTPUT_FILE = "data/processed/segmentation_dataset.jsonl"
LIMIT = 200000  # Increased limit for "Pro" training

# STRICTER PATTERNS (The "Strict Teacher")
PATTERNS = {
    "ORDER": [
        r"bail is granted", r"application allowed", r"released on bail",
        r"petition dismissed", r"accordingly disposed", r"interim bail",
        r"subject to the following conditions", r"bond of rs"
    ],
    "ARGUMENTS": [
        r"learned counsel", r"submitted that", r"argued that", 
        r"submission of", r"contended", r"urged", r"opposed by",
        r"public prosecutor"
    ],
    "REASONING": [
        r"court is of the opinion", r"having heard", r"considering the facts",
        r"perusal of", r"prima facie", r"totality of circumstances",
        r"without expressing any opinion"
    ],
    "FACTS": [
        r"prosecution case", r"allegation is", r"fir no", 
        r"registered under", r"complainant stated", r"accused was arrested"
    ]
}

def label_text(text):
    segments = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if len(line) < 40: continue # Skip short noise
        
        lower = line.lower()
        label = "FACTS" # Default
        
        # Priority Check (Order > Reasoning > Arguments > Facts)
        if any(r in lower for r in PATTERNS["ORDER"]): label = "ORDER"
        elif any(r in lower for r in PATTERNS["REASONING"]): label = "REASONING"
        elif any(r in lower for r in PATTERNS["ARGUMENTS"]): label = "ARGUMENTS"
        elif any(r in lower for r in PATTERNS["FACTS"]): label = "FACTS"
        
        segments.append({"text": line, "label": label})
    return segments

def main():
    print("🚀 GENERATING PRO SEGMENTATION DATASET...")
    count = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
            for line in tqdm(f_in):
                if count >= LIMIT: break
                try:
                    data = json.loads(line)
                    segs = label_text(data.get('full_text', ''))
                    for s in segs:
                        f_out.write(json.dumps(s) + "\n")
                    count += 1
                except: continue
    print(f"✅ Data Ready: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()