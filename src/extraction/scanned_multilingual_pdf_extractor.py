#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import tempfile
import json
import concurrent.futures
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# Third-party libraries
from pdf2image import convert_from_path
import pytesseract

from rich.console import Console

console = Console()

# --- Configuration and Environment Variables ---

# Pathing: Calculates the project root (parents[2]) and defines base directories from there.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PDF_BASE_DIR: Path = PROJECT_ROOT / 'data' / 'pdfs'
CLASSIFICATION_REPORT_PATH: Path = PROJECT_ROOT / 'data' / 'classification_report.json' # Adjusted based on your tree
OUTPUT_BASE_DIR: Path = PROJECT_ROOT / 'data' / 'processed' / 'extracted_md'

# Mapping of language folder codes to Tesseract language strings.
# We include 'eng' in all to handle the "very little bit of English" in the documents.
TESSERACT_LANG_MAP: Dict[str, str] = {
    'ur': 'urd+eng',      # Urdu
    'bn': 'ben+eng',      # Bengali
    'zh': 'chi_sim+eng',  # Simplified Chinese
}

# --- Utility Functions ---

def load_classification_report(json_path: Path) -> Optional[Dict[str, Any]]:
    """Loads and parses the classification report JSON file."""
    if not json_path.exists():
        print(f"FATAL ERROR: Classification report not found at {json_path}")
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"FATAL ERROR: Could not load or parse classification report: {e}")
        return None

def preprocess_and_ocr(pdf_path: Path, lang_code: str) -> Tuple[str, Optional[str]]:
    """
    Performs PDF page to image conversion and OCR for a single PDF file.
    Designed to run in parallel processes.
    
    Returns: A tuple of (original_pdf_filename, extracted_text).
    """
    filename: str = pdf_path.name
    tesseract_langs: str = TESSERACT_LANG_MAP.get(lang_code, 'eng')
    print(f"   [Process {os.getpid()}] Starting OCR for {filename} (Lang: {lang_code})...")
    
    # Use a temporary directory local to this process for efficiency
    with tempfile.TemporaryDirectory() as _:
        try:
            # 1. PDF Page to Image Conversion (High DPI for better OCR)
            # This converts all pages of the PDF into a list of high-res PIL images.
            images: List[Any] = convert_from_path(
                str(pdf_path), 
                dpi=400, # High DPI is critical for scanned documents
                fmt='png',
                thread_count=1
            )
            
            if not images:
                print(f"   [Process {os.getpid()}] WARNING: {filename} has no pages or failed conversion.")
                return filename, None

            all_raw_text: List[str] = []
            
            # 2. OCR Execution (Page by Page)
            for i, image in enumerate(images):
                # We can save and load the image here, but Tesseract can often take the PIL image directly.
                # Keeping the direct image passing is often more efficient.
                raw_text: str = pytesseract.image_to_string(
                    image, 
                    lang=tesseract_langs,
                    config='--psm 3' # PSM 3 is usually best for a single column page
                )
                all_raw_text.append(raw_text.strip())
            
            # Combine all pages with a clear page break marker
            full_raw_text: str = "\n\n--- PAGE BREAK ---\n\n".join(all_raw_text)
            
            # Simple cleanup of common OCR errors (e.g., excessive newlines)
            cleaned_text: str = re.sub(r'\n\s*\n', '\n\n', full_raw_text).strip()
            
            print(f"   [Process {os.getpid()}] {filename} OCR finished. Extracted {len(cleaned_text.split())} words.")
            
            return filename, cleaned_text
            
        except pytesseract.TesseractNotFoundError:
            print("\nFATAL ERROR: Tesseract executable not found. Please install Tesseract OCR.")
            return filename, None
        except Exception as e:
            print(f"\nFATAL ERROR: {filename} processing failed: {e}")
            return filename, None

def save_markdown_output(filename: str, lang_code: str, extracted_text: str) -> None:
    """Saves the extracted text to the required Markdown destination."""
    
    # 1. Define output structure and ensure it exists
    output_dir: Path = OUTPUT_BASE_DIR / lang_code
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Create the output file path (.pdf -> .md)
    markdown_filename: str = filename.replace('.pdf', '.md')
    output_path: Path = output_dir / markdown_filename
    
    # 3. Save the content
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        print(f"  -> SUCCESS: Saved {markdown_filename} to {output_dir.relative_to(PROJECT_ROOT)}")
    except Exception as e:
        print(f"  -> ERROR: Failed to save file {output_path}: {e}")


# --- Main Execution Flow ---

def run_extraction_pipeline() -> None:
    """
    Orchestrates the entire parallel OCR process based on the classification report.
    """
    console.print("🚀 [bold cyan]Starting High-Performance Multilingual OCR Pipeline for Scanned PDFs[/bold cyan]")
    
    # 1. LOAD CLASSIFICATION REPORT
    report: Optional[Dict[str, Any]] = load_classification_report(CLASSIFICATION_REPORT_PATH)
    if not report or not report.get('results'):
        console.print("[red]Pipeline aborted: Cannot load or process classification report.[/red]")
        return
        
    # 2. GATHER ALL TASKS
    all_tasks: List[Tuple[Path, str]] = [] # List of (pdf_path, lang_code)
    
    print("🔍 Analyzing classification report for scanned PDFs...")
    for lang_code, lang_data in report['results'].items():
        if lang_code not in TESSERACT_LANG_MAP:
            console.print(f"[red]WARNING[/red]: Skipping language code '{lang_code}'. Tesseract model not mapped.")
            continue
            
        lang_folder: Path = PDF_BASE_DIR / lang_code
        
        # Check for files flagged as scanned
        for filename in lang_data.get('scanned_pdfs', []):
            pdf_path: Path = lang_folder / filename
            if pdf_path.exists():
                all_tasks.append((pdf_path, lang_code))
            else:
                console.print(f"[red]WARNING[/red]: File listed in report not found: {pdf_path}")

    if not all_tasks:
        print("✅ No scanned PDFs found to process. Pipeline finished.")
        return

    print(f"✅ Found {len(all_tasks)} scanned PDFs to process. Starting parallel OCR.")

    # 3. PARALLEL OCR EXECUTION (CPU-BOUND)
    
    # Use as many workers as available CPU cores for maximum OCR throughput
    MAX_WORKERS: int = os.cpu_count() or 4 
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        
        # Dictionary to track futures: {Future object: (original_filename, lang_code)}
        future_to_file: Dict[concurrent.futures.Future, Tuple[str, str]] = {}
        
        for pdf_path, lang_code in all_tasks:
            # Submitting the task to the pool. We pass the full path and the language code.
            future = executor.submit(preprocess_and_ocr, pdf_path, lang_code)
            future_to_file[future] = (pdf_path.name, lang_code)

        # Process results as they complete (as_completed for max throughput)
        for future in concurrent.futures.as_completed(future_to_file):
            filename, lang_code = future_to_file[future]
            try:
                # Result is (filename, extracted_text)
                _, extracted_text = future.result() 
                
                if extracted_text:
                    # 4. SAVE OUTPUT
                    save_markdown_output(filename, lang_code, extracted_text)
                else:
                    console.print(f"[red]WARNING[/red]: OCR failed or returned empty for {filename}.")
                    
            except Exception as exc:
                console.print(f"[red]ERROR[/red]: {filename} generated an exception: {exc}")

    print("")
    console.print("🎉 [green]SUCCESS[/green]: Finished processing all specified scanned PDFs.")



