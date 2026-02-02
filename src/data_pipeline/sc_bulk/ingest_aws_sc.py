import os
import json
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import pandas as pd
import io
from tqdm import tqdm
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
BUCKET_NAME = "indian-supreme-court-judgments"
OUTPUT_FILE = "data/processed/aws_sc_data.jsonl"
YEARS = [2021, 2022, 2023, 2024]

def clean_html_text(html_content):
    """Converts raw HTML to clean text."""
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text(separator="\n").strip()
    except:
        return ""

def main():
    print(f"🚀 Connecting to AWS SC Bucket: {BUCKET_NAME}...")
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="ap-south-1")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    total_added = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for year in YEARS:
            prefix = f"metadata/parquet/year={year}/"
            try:
                response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
                if 'Contents' not in response: continue
                
                files = [x['Key'] for x in response['Contents'] if x['Key'].endswith('.parquet')]
                print(f"   📅 Fetching SC {year} ({len(files)} files)...")
                
                for key in tqdm(files):
                    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
                    df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
                    
                    # TARGET THE RAW_HTML COLUMN
                    if 'raw_html' not in df.columns:
                        continue

                    # Filter: Drop empty rows
                    df = df.dropna(subset=['raw_html'])
                    
                    # Filter: Check keywords directly in HTML (faster)
                    keywords = 'bail|criminal|murder|custody|detention|302|ipc|crpc'
                    mask = df['raw_html'].str.contains(keywords, case=False, na=False)
                    relevant_df = df[mask]
                    
                    for _, row in relevant_df.iterrows():
                        # CLEAN THE HTML
                        raw_text = row['raw_html']
                        clean_text = clean_html_text(raw_text)
                        
                        if len(clean_text) < 200: continue 

                        entry = {
                            "court": "Supreme Court (AWS)",
                            "year": year,
                            "full_text": clean_text, # Saved as clean text
                            "outcome": "UNCLEAR",
                            "source": "AWS_SC_Vanga",
                            "instruction": "Analyze this Supreme Court judgment.",
                            "output": "Analysis provided."
                        }
                        f_out.write(json.dumps(entry) + "\n")
                        total_added += 1
                        
            except Exception as e:
                print(f"   ⚠️ Error processing {year}: {e}")

    print(f"\n✅ SC INGESTION COMPLETE.")
    print(f"   - Added {total_added} cases.")
    print(f"   - Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()