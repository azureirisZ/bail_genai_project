import os
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))

META_DIR = os.path.join(PROJECT_ROOT, "data", "metadata")
os.makedirs(META_DIR, exist_ok=True)

SC_METADATA_URL = (
    "https://indian-supreme-court-judgments.s3.amazonaws.com/"
    "metadata/judgments_metadata.csv"
)

OUT_FILE = os.path.join(META_DIR, "sc_metadata.csv")

def main():
    if os.path.exists(OUT_FILE):
        print("✅ Metadata already downloaded.")
        return

    print("⬇️ Downloading Supreme Court metadata...")
    urllib.request.urlretrieve(SC_METADATA_URL, OUT_FILE)
    print(f"✅ Saved to {OUT_FILE}")

if __name__ == "__main__":
    main()
