#!/usr/bin/env python3
"""
Extract text from all PDFs and markdown files in raw/ directory.
Outputs a single training text file for llama-finetune.
"""
import subprocess
import os
import glob
import sys
import re

RAW_DIR = "/home/obsidian/mike/raw"
OUTPUT_DIR = "/home/model/training_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "training_corpus.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    """Clean extracted text - remove excessive whitespace, page numbers, etc."""
    # Remove form feeds
    text = text.replace('\f', '\n')
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove lines that are just page numbers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Remove common PDF artifacts
    text = re.sub(r'^\s*Authorized licensed use limited to:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*Downloaded on.*$', '', text, flags=re.MULTILINE)
    # Strip trailing whitespace per line
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    return text.strip()

def extract_pdf(pdf_path):
    """Extract text from a PDF using pdftotext."""
    try:
        result = subprocess.run(
            ['pdftotext', '-layout', pdf_path, '-'],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0 and result.stdout.strip():
            return clean_text(result.stdout)
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  ERROR: {e}", file=sys.stderr)
    return None

def read_markdown(md_path):
    """Read a markdown file."""
    try:
        with open(md_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read().strip()
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
    return None

# Collect all files
pdf_files = sorted(glob.glob(os.path.join(RAW_DIR, "**/*.pdf"), recursive=True))
md_files = sorted(glob.glob(os.path.join(RAW_DIR, "**/*.md"), recursive=True))

print(f"Found {len(pdf_files)} PDFs and {len(md_files)} markdown files")
print(f"Output: {OUTPUT_FILE}")

total_chars = 0
total_docs = 0
errors = 0

with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
    # Process PDFs
    for i, pdf_path in enumerate(pdf_files):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"[{i+1}/{len(pdf_files)}] Processing PDFs...")
        
        text = extract_pdf(pdf_path)
        if text and len(text) > 100:  # Skip very short extractions
            # Add document separator with source info
            rel_path = os.path.relpath(pdf_path, RAW_DIR)
            out.write(f"\n\n### Document: {rel_path}\n\n")
            out.write(text)
            out.write("\n")
            total_chars += len(text)
            total_docs += 1
        else:
            errors += 1
    
    # Process markdown files
    for i, md_path in enumerate(md_files):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[{i+1}/{len(md_files)}] Processing markdown...")
        
        text = read_markdown(md_path)
        if text and len(text) > 50:
            rel_path = os.path.relpath(md_path, RAW_DIR)
            out.write(f"\n\n### Document: {rel_path}\n\n")
            out.write(text)
            out.write("\n")
            total_chars += len(text)
            total_docs += 1
        else:
            errors += 1

print(f"\nDone!")
print(f"  Documents processed: {total_docs}")
print(f"  Errors/skipped: {errors}")
print(f"  Total characters: {total_chars:,}")
print(f"  Output size: {os.path.getsize(OUTPUT_FILE) / 1e6:.1f} MB")
print(f"  Output: {OUTPUT_FILE}")
