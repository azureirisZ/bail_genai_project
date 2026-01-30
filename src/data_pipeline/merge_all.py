import os

# --- CONFIGURATION ---
SOURCES = [
    "data/processed/aws_sc_data.jsonl",       # Supreme Court (~1,800)
    "data/processed/scraped_hc_data.jsonl",   # Scraped High Court (~600)
    "data/processed/aws_hc_bail_data.jsonl"   # AWS High Court (~626,000)
]
FINAL_OUTPUT = "data/processed/final_training_data.jsonl"

def main():
    print("🔄 MERGING ALL DATASETS (SC + HC + AWS)...")
    total_count = 0
    
    with open(FINAL_OUTPUT, 'w', encoding='utf-8') as f_out:
        for source in SOURCES:
            if os.path.exists(source):
                print(f"   ➕ Adding: {source}...")
                with open(source, 'r', encoding='utf-8') as f_in:
                    for line in f_in:
                        if line.strip():
                            f_out.write(line)
                            total_count += 1
            else:
                print(f"   ⚠️ Source not found: {source}")
    
    print(f"\n✅ FINAL DATASET READY: {total_count} cases")
    print(f"   Saved to: {FINAL_OUTPUT}")

if __name__ == "__main__":
    main()