import os
import json
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import pandas as pd
import io
from tqdm import tqdm

# --- CONFIGURATION ---
BUCKET_NAME = "indian-high-court-judgments"
OUTPUT_FILE = "data/processed/aws_hc_bail_data.jsonl"
TEMP_DIR = "data/interim/aws_hc"

# We only want recent data (2022-2024) to keep it manageable
# The bucket is structured as: metadata/parquet/year=YYYY/
YEARS_TO_FETCH = [2022, 2023, 2024]

# Keywords to keep (Criminal Law only)
KEYWORDS = [
    "bail", "anticipatory", "parole", "custody", "detention",
    "ndps", "uapa", "pmla", "murder", "rape", "302 ipc", "section 37"
]

def get_s3_client():
    """Connects to AWS S3 without credentials (Public Bucket)."""
    return boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="ap-south-1")

def main():
    print(f"🚀 Connecting to AWS Bucket: {BUCKET_NAME}...")
    s3 = get_s3_client()
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    total_cases = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for year in YEARS_TO_FETCH:
            print(f"\n📅 Scanning Year: {year}...")
            
            # 1. List files in the year folder
            prefix = f"metadata/parquet/year={year}/"
            
            try:
                response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
                if 'Contents' not in response:
                    print(f"   ⚠️ No data found for {year}.")
                    continue
                
                # 2. Find Parquet files (There might be multiple parts)
                parquet_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith(".parquet")]
                
                print(f"   found {len(parquet_files)} metadata files. Downloading & Filtering...")
                
                for pq_file in tqdm(parquet_files):
                    try:
                        # Download to memory
                        obj = s3.get_object(Bucket=BUCKET_NAME, Key=pq_file)
                        df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
                        
                        # Check if text column exists (usually 'raw_text' or 'judgment_text')
                        # In vanga's SC dataset it was 'raw_html' or 'text'
                        text_col = None
                        for col in df.columns:
                            if 'text' in col.lower() or 'html' in col.lower() or 'content' in col.lower():
                                text_col = col
                                break
                        
                        if not text_col: continue

                        # FILTERING
                        df = df.dropna(subset=[text_col])
                        mask = df[text_col].str.contains('|'.join(KEYWORDS), case=False, regex=True)
                        relevant_df = df[mask]
                        
                        if relevant_df.empty: continue

                        # SAVE
                        for _, row in relevant_df.iterrows():
                            text = row[text_col]
                            
                            # Heuristic Outcome
                            outcome = "UNCLEAR"
                            lower_text = str(text).lower()[-1000:]
                            if "allowed" in lower_text or "granted" in lower_text:
                                outcome = "BAIL_GRANTED"
                            elif "dismissed" in lower_text or "rejected" in lower_text:
                                outcome = "BAIL_DENIED"

                            data_point = {
                                "court": "High Court (AWS)",
                                "year": year,
                                "full_text": str(text),
                                "outcome": outcome,
                                "source": "AWS_OpenData_Vanga",
                                "instruction": "Analyze this High Court judgment and determine the relief granted.",
                                "output": f"Outcome: {outcome}."
                            }
                            
                            f_out.write(json.dumps(data_point) + "\n")
                            total_cases += 1
                            
                    except Exception as e:
                        # Sometimes a single parquet file is corrupt, just skip it
                        continue
                        
            except Exception as e:
                print(f"   ❌ Error accessing {year}: {e}")

    print(f"\n✅ AWS HIGH COURT INGESTION COMPLETE.")
    print(f"   - Total New Cases: {total_cases}")
    print(f"   - Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()