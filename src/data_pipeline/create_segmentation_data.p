import json
from tqdm import tqdm

INPUT_FILE = "data/processed/final_training_data.jsonl"
OUTPUT_FILE = "data/processed/segmentation_dataset.jsonl"
# NO LIMIT. WE PROCESS EVERYTHING.

PATTERNS = {
    "ORDER": [r"bail is granted", r"application allowed", r"released on bail", r"petition dismissed"],
    "ARGUMENTS": [r"learned counsel", r"submitted that", r"argued that", r"contended"],
    "REASONING": [r"court is of the opinion", r"having heard", r"considering the facts"],
    "FACTS": [r"prosecution case", r"allegation is", r"fir no", r"accused was arrested"]
}

def main():
    print("🚀 GENERATING FULL DATASET (600k+ Cases)...")
    count = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
            for line in tqdm(f_in):
                try:
                    data = json.loads(line)
                    text = data.get('full_text', '')
                    lines = text.split('\n')
                    for line in lines:
                        if len(line) < 40: continue
                        label = "FACTS"
                        lower = line.lower()
                        if any(x in lower for x in PATTERNS["ORDER"]): label = "ORDER"
                        elif any(x in lower for x in PATTERNS["REASONING"]): label = "REASONING"
                        elif any(x in lower for x in PATTERNS["ARGUMENTS"]): label = "ARGUMENTS"
                        f_out.write(json.dumps({"text": line.strip(), "label": label}) + "\n")
                    count += 1
                except: continue
    print(f"✅ Full Dataset Ready: {count} cases processed.")

if __name__ == "__main__":
    main()