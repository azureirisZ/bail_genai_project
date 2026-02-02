import os
import glob
import argparse
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def parse_html_to_text(file_info):
    """
    Worker function to convert a single HTML file to text.
    """
    input_path, output_path = file_info
    
    try:
        # 1. Read HTML
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 2. Extract Text (BeautifulSoup)
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove scripts and styles to clean up the output
        for script in soup(["script", "style"]):
            script.extract()
            
        text = soup.get_text(separator='\n')

        # 3. Basic Whitespace cleanup
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = '\n'.join(chunk for chunk in chunks if chunk)

        # 4. Save to TXT
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(clean_text)
            
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Bulk convert HTML to Text")
    parser.add_argument("--input", default="data/raw/hc_bulk", help="Input directory containing .html files")
    parser.add_argument("--output", default="data/interim/hc_bulk_txt", help="Output directory for .txt files")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Gather all HTML files
    print(f"🔍 Scanning {args.input} for .html files...")
    search_pattern = os.path.join(args.input, "**", "*.html")
    # recursive=True ensures we find files even if they are in subfolders
    html_files = glob.glob(search_pattern, recursive=True)
    
    if not html_files:
        print(f"❌ No HTML files found in {args.input}")
        return

    print(f"✅ Found {len(html_files)} files. Starting conversion...")

    # Prepare tasks (Input Path -> Output Path)
    tasks = []
    for html_file in html_files:
        # Create a mirrored filename with .txt extension
        base_name = os.path.basename(html_file)
        file_name_no_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(args.output, f"{file_name_no_ext}.txt")
        tasks.append((html_file, output_file))

    # Run in parallel
    success_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(parse_html_to_text, tasks), total=len(tasks), unit="doc"))
        success_count = sum(results)

    print(f"🎉 Conversion Complete. Successfully processed {success_count}/{len(tasks)} files.")
    print(f"📂 Text files saved to: {args.output}")

if __name__ == "__main__":
    main()