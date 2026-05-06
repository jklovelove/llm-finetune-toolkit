#!/usr/bin/env python3
"""
Fast batch PDF to Markdown conversion using pdftotext.
Much faster than marker for large collections.
"""
import os
import sys
import glob
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def convert_pdf(pdf_path, output_dir):
    """Convert a single PDF to markdown using pdftotext."""
    try:
        md_name = Path(pdf_path).stem + ".md"
        md_path = os.path.join(output_dir, md_name)
        
        # Skip if exists
        if os.path.exists(md_path):
            return pdf_path, True, "Skipped (exists)"
        
        result = subprocess.run(
            ['pdftotext', '-layout', pdf_path, '-'],
            capture_output=True, text=True, timeout=60
        )
        
        if result.returncode == 0 and result.stdout.strip():
            text = result.stdout.strip()
            # Clean up
            text = text.replace('\f', '\n')
            import re
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
            text = '\n'.join(line.rstrip() for line in text.split('\n'))
            text = text.strip()
            
            if len(text) > 100:
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(f"# {Path(pdf_path).stem}\n\n{text}\n")
                return pdf_path, True, "OK"
            else:
                return pdf_path, False, "Too short"
        else:
            return pdf_path, False, "pdftotext failed"
    except subprocess.TimeoutExpired:
        return pdf_path, False, "Timeout"
    except Exception as e:
        return pdf_path, False, str(e)[:100]

def main():
    parser = argparse.ArgumentParser(description="Fast PDF to Markdown conversion")
    parser.add_argument("--input", required=True, help="Input directory with PDFs")
    parser.add_argument("--output", required=True, help="Output directory for MDs")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--max-files", type=int, default=0, help="Max files to process (0=all)")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Find all PDFs
    pdfs = sorted(glob.glob(os.path.join(args.input, "**/*.pdf"), recursive=True))
    print(f"Found {len(pdfs)} PDFs in {args.input}")
    
    if args.max_files > 0:
        pdfs = pdfs[:args.max_files]
        print(f"Processing first {args.max_files} files")
    
    print(f"Processing {len(pdfs)} PDFs with {args.workers} workers...")
    
    success = 0
    failed = 0
    skipped = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(convert_pdf, pdf, args.output): pdf 
            for pdf in pdfs
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            pdf_path, ok, msg = future.result()
            if ok:
                if "Skipped" in msg:
                    skipped += 1
                else:
                    success += 1
            else:
                failed += 1
                if failed <= 10:  # Only print first 10 failures
                    print(f"  FAIL: {os.path.basename(pdf_path)} - {msg}")
            
            if i % 100 == 0:
                print(f"  [{i}/{len(pdfs)}] OK: {success}, Skip: {skipped}, Fail: {failed}")
    
    print(f"\nDone! Success: {success}, Skipped: {skipped}, Failed: {failed}")
    print(f"Output: {args.output}")

if __name__ == "__main__":
    main()
