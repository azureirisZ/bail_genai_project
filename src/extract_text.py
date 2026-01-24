"""
Step 1: PDF Text Extraction for Bail Judgments
---------------------------------------------

This script:
1. Loads a bail order PDF
2. Extracts raw text page-by-page
3. Cleans headers, footers, and noise
4. Saves cleaned text to a .txt file

This is the foundation for:
- Legal segmentation
- Factor extraction
- Similarity modeling
"""

import pdfplumber
import re
from pathlib import Path

# --------- PATHS ---------

PDF_PATH = Path("../data/raw_pdfs/sample_case.pdf")
OUTPUT_PATH = Path("../data/extracted_text/sample_case.txt")

# --------- CLEANING FUNCTION ---------

def clean_text(text: str) -> str:
    """
    Removes common noise from Indian court judgments
    """
    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Remove page numbers like "Page 1 of 12"
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)

    # Remove common header/footer patterns
    text = re.sub(r"SUPREME COURT OF INDIA", "", text, flags=re.IGNORECASE)
    text = re.sub(r"REPORTABLE|NON-REPORTABLE", "", text, flags=re.IGNORECASE)

    return text.strip()

# --------- EXTRACTION PIPELINE ---------

def extract_pdf_text(pdf_path: Path) -> str:
    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                cleaned = clean_text(page_text)
                full_text.append(cleaned)

    return "\n\n".join(full_text)

# --------- MAIN ---------

if __name__ == "__main__":
    print("📄 Extracting text from PDF...")

    text = extract_pdf_text(PDF_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    print("✅ Extraction complete!")
    print(f"📁 Saved to: {OUTPUT_PATH.resolve()}")
