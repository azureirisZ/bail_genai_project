import json
import os
from collections import Counter

# --- CONFIGURATION ---
DATA_FILE = "data/processed/bail_dataset_v1.jsonl"

def main():
    if not os.path.exists(DATA_FILE):
        print(f"❌ File not found: {DATA_FILE}")
        print("Did you run normalize.py?")
        return

    print(f"📊 Analyzing {DATA_FILE}...")
    
    total = 0
    outcomes = []
    statutes = []
    
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            total += 1
            
            # Check Outcome
            if data['outcome'] == "BAIL_GRANTED":
                outcomes.append("GRANTED")
            elif data['outcome'] == "BAIL_DENIED":
                outcomes.append("DENIED")
            else:
                outcomes.append("UNCLEAR")
            
            # Check Statute (IPC, NDPS, etc.)
            statutes.append(data.get('statute', 'Unknown'))

    # PRINT REPORT
    print(f"\n✅ Total Cases: {total}")
    
    print("\n⚖️  OUTCOME BALANCE:")
    counts = Counter(outcomes)
    for k, v in counts.items():
        print(f"   - {k}: {v} ({(v/total)*100:.1f}%)")

    print("\n📜 STATUTE DISTRIBUTION:")
    s_counts = Counter(statutes)
    for k, v in s_counts.most_common():
        print(f"   - {k}: {v}")

    # VERDICT
    if counts['GRANTED'] < 50 or counts['DENIED'] < 50:
        print("\n⚠️  WARNING: Serious Class Imbalance!")
        print("   -> We might need to fetch more data for the minority class.")
    else:
        print("\n🟢 DATASET HEALTHY. Ready for Training.")

if __name__ == "__main__":
    main()