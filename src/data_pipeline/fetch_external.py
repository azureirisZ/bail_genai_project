import os
import time
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURATION ---
OUTPUT_DIR = "data/raw/hc_bulk"
TARGET_COUNT = 2000  # How many extra cases do you want?

def main():
    print("🚀 Connecting to Hugging Face (Official ILDC)...")
    
    try:
        # 1. Load in STREAMING mode (Instant start, no huge download)
        # We use 'ildc_multi' to get judgments from multiple courts
        dataset = load_dataset(
            "opennyai/ildc", 
            "ildc_multi", 
            split="train", 
            streaming=True, 
            trust_remote_code=True
        )
        
        print(f"✅ Connected! Fetching {TARGET_COUNT} cases...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        count = 0
        
        # 2. Iterate through the stream
        for i, row in tqdm(enumerate(dataset), total=TARGET_COUNT):
            if count >= TARGET_COUNT:
                break
                
            text = row.get('text')
            
            # Basic validation
            if text and len(text) > 500: # Skip tiny fragments
                
                filename = f"External_ILDC_{i}.html"
                filepath = os.path.join(OUTPUT_DIR, filename)
                
                # Don't overwrite existing work
                if os.path.exists(filepath):
                    continue
                
                # 3. WRAP IN FAKE HTML
                # This tricks your pipeline into thinking these are scraped web pages
                fake_html = f"""
                <html>
                    <head><title>External ILDC Case {i}</title></head>
                    <body>
                        <div class="judgments">
                            <pre>{text}</pre>
                        </div>
                    </body>
                </html>
                """
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(fake_html)
                
                count += 1
                
        print(f"🎉 Done! Successfully added {count} high-quality legal cases.")
        print("👉 These are saved as .html files in 'data/raw/hc_bulk'")
        print("👉 You can now run 'html_to_text.py' to process them.")

    except Exception as e:
        print("\n❌ ERROR DETAILS:")
        print(e)
        print("\n⚠️ TROUBLESHOOTING:")
        print("1. If it says 'trust_remote_code', make sure that flag is True (it is in this script).")
        print("2. If it says 'Connection Error', check your internet.")

if __name__ == "__main__":
    main()