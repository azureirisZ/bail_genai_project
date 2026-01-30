import os
import json
import re
import glob
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_DIR = "data/interim/hc_bulk_txt"
OUTPUT_FILE = "data/processed/bail_dataset_v1.jsonl"

# --- RULES & KEYWORDS ---
# Layer 1: The Statute
STATUTES = {
    "NDPS": ["ndps", "narcotic", "psychotropic", "contraband", "ganja", "opium"],
    "POCSO": ["pocso", "protection of children", "sexual offence", "minor victim"],
    "PMLA": ["pmla", "money laundering", "proceeds of crime", "enforcement directorate"],
    "UAPA": ["uapa", "unlawful activities", "terrorist", "nia act"],
    "IPC": ["ipc", "indian penal code", "murder", "theft", "assault"]
}

# Layer 2: The Merits (Why was it granted/denied?)
FACTORS = {
    "delay_in_trial": ["long incarceration", "trial has not commenced", "delay in trial", "custody for"],
    "parity": ["co-accused", "parity", "similarly situated"],
    "medical_grounds": ["medical condition", "illness", "treatment", "hospital"],
    "false_implication": ["falsely implicated", "political rivalry", "prior enmity"],
    "compromise": ["compromise", "settlement", "amicably settled"],
    # Negative Factors
    "criminal_history": ["criminal antecedent", "past record", "habitual offender", "history sheet"],
    "flight_risk": ["abscond", "flight risk", "evade process"],
    "tampering": ["tampering with evidence", "influence witnesses", "threaten witnesses"],
    # Specifics
    "commercial_quantity": ["commercial quantity", "small quantity", "intermediate quantity"],
    "child_victim": ["minor victim", "child", "age of the victim"]
}

def clean_text(text):
    """Basic cleanup to make regex easier."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text) # Remove extra whitespace
    return text

def determine_outcome(text):
    """
    Heuristic to determine if bail was Granted or Denied.
    We look at the LAST 500 characters of the judgment usually.
    """
    # Grab the end of the judgment where the order is passed
    conclusion = text[-1000:] 
    
    # strong positive signals
    if any(x in conclusion for x in ["bail granted", "allowed", "enlarged on bail", "released on bail", "application is allowed"]):
        return "BAIL_GRANTED"
    
    # strong negative signals
    if any(x in conclusion for x in ["dismissed", "rejected", "devoid of merit", "cant be granted", "not inclined to grant"]):
        return "BAIL_DENIED"
    
    return "UNCLEAR"

def extract_statute(text):
    """Finds the primary law involved."""
    scores = {k: 0 for k in STATUTES}
    for stat, keywords in STATUTES.items():
        for kw in keywords:
            if kw in text:
                scores[stat] += 1
    
    # Return the statute with max hits, or "Other"
    best_match = max(scores, key=scores.get)
    return best_match if scores[best_match] > 0 else "IPC" # Default to IPC

def extract_factors(text):
    """Checks for presence of specific legal arguments."""
    present_factors = []
    for factor, keywords in FACTORS.items():
        if any(kw in text for kw in keywords):
            present_factors.append(factor)
    return present_factors

def main():
    print("🧠 Starting Normalization (Extracting Legal Logic)...")
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    if not files:
        print(f"❌ No text files found in {INPUT_DIR}. Did you run html_to_text.py?")
        return

    processed_count = 0
    skipped_count = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for filepath in tqdm(files):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    raw_content = f.read()
                
                # 1. Clean
                clean_content = clean_text(raw_content)
                if len(clean_content) < 200: # Skip empty/junk files
                    continue

                # 2. Extract Logic
                outcome = determine_outcome(clean_content)
                if outcome == "UNCLEAR":
                    skipped_count += 1
                    continue # Skip cases where we can't tell the result
                
                statute = extract_statute(clean_content)
                legal_factors = extract_factors(clean_content)
                
                # 3. Structure Data for LLM
                # This is the "Prompt" -> "Completion" format
                data_point = {
                    "filename": os.path.basename(filepath),
                    "full_text": raw_content, # We keep raw text for training
                    "statute": statute,
                    "outcome": outcome,
                    "factors": legal_factors,
                    
                    # INSTRUCTION TUNING FORMAT (for later)
                    "instruction": f"Analyze this {statute} bail application. Identify key factors and predict the outcome.",
                    "output": f"Outcome: {outcome}. Key Factors: {', '.join(legal_factors)}."
                }
                
                f_out.write(json.dumps(data_point) + "\n")
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    print(f"\n✅ Processing Complete.")
    print(f"   - Saved: {processed_count} cases")
    print(f"   - Skipped (Unclear Outcome): {skipped_count} cases")
    print(f"   - Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()