#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import json
from pathlib import Path
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple

import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader
from rich.console import Console


console = Console()

# --- Configuration and Environment Variables ---
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PDF_BASE_DIR: Path = PROJECT_ROOT / 'data' / 'pdfs'
CLASSIFICATION_REPORT_PATH: Path = PROJECT_ROOT / 'data' / 'classification_report.json'
OUTPUT_BASE_DIR: Path = PROJECT_ROOT / 'data' / 'processed' / 'extracted_md'

# Tesseract Language Map (only used for the 'ur' OCR exception)
TESSERACT_LANG_MAP: Dict[str, str] = {
    'ur': 'urd+eng',     # Urdu + English fallback for better accuracy
    'bn': 'ben+eng',
    'zh': 'chi_sim+eng',
    'default': 'eng'
}

# SPECIAL RULE: Specific files that require constrained extraction (ignore page 1, process next 10 pages).
# Page numbers here are 1-based: Start Page (inclusive), End Page (inclusive).
# Start at page 2, process 10 pages -> ends at page 11.
SPECIAL_PAGE_CONFIG = {
    "shora e rampur.pdf": {'start_page': 2, 'end_page': 30},
    "fasana-e-ajaib final.pdf": {'start_page': 2, 'end_page': 30},
    # "Research Nirdeshika.pdf": {'start_page': 2, 'end_page': 11},
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

def _clean_extracted_text(text: str) -> str:
    """
    Performs cleanup to standardize formatting and heuristically filter out 
    content that looks like page numbers, line numbers, or spurious metadata 
    from the digital extraction process.
    """
    if not text:
        return ""

    # 1. Standardize whitespace
    cleaned_text = re.sub(r'[ \t]+', ' ', text) 
    # Consolidate multiple newlines
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text).strip() 
    
    # 2. Filter out potential metadata (page numbers, line numbers, etc.)
    lines = cleaned_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped_line = line.strip()
        
        # Criterion: Line is extremely short (<= 5 chars) AND contains only digits/punctuation. 
        # This prevents filtering actual short sentences in Urdu/Bengali/Chinese.
        # \u0600-\u06FF (Arabic/Urdu), \u0980-\u09FF (Bengali), \u4E00-\u9FFF (CJK Unified)
        if len(stripped_line) <= 5 and not re.search(r'[a-zA-Z\u0600-\u06FF\u0980-\u09FF\u4E00-\u9FFF]', stripped_line):
            continue
        
        if not stripped_line:
            continue
            
        cleaned_lines.append(line)
        
    final_text: str = "\n".join(cleaned_lines)
    final_text = re.sub(r'\n\s*\n', '\n\n', final_text).strip()
    
    return final_text

def extract_digital_pdf_text(
    pdf_path: Path, 
    lang_code: str, 
    start_page: int = 1, # 1-based index to start extraction (inclusive)
    end_page: Optional[int] = None # 1-based index to end extraction (inclusive)
) -> Tuple[str, Optional[str]]:
    """
    Parses text directly from a digital PDF file using pypdf.
    """
    filename: str = pdf_path.name
    print(f"   [Process {os.getpid()}] Starting DIGITAL extraction for {filename} (Lang: {lang_code})...")
    
    if PdfReader is None:
        return filename, None
        
    all_raw_text: List[str] = []
    
    try:
        reader = PdfReader(str(pdf_path))
        num_pages = len(reader.pages)

        # Calculate 0-based indices for Python's range function
        start_index = max(0, start_page - 1)
        # range is exclusive on the end, so if end_page is 11, the loop should go up to 10 (index 10)
        end_index_exclusive = num_pages if end_page is None else min(num_pages, end_page) 
        
        # Check for valid range
        if start_index >= end_index_exclusive:
            print(f"   [Process {os.getpid()}] WARNING: Invalid page range: Pages {start_page} to {end_page} are out of bounds or empty for {filename}.")
            return filename, None

        print(f"   [Process {os.getpid()}] Processing digital pages {start_page} to {end_index_exclusive}...")

        for i in range(start_index, end_index_exclusive):
            page = reader.pages[i]
            page_text = page.extract_text()
            
            # Filter out pages that return only white space (likely pure image pages)
            if page_text and page_text.strip():
                all_raw_text.append(page_text)
        
        if not all_raw_text:
            print(f"   [Process {os.getpid()}] WARNING: {filename} contained no readable digital text in the range.")
            return filename, None
            
        full_raw_text: str = "\n\n--- PAGE BREAK ---\n\n".join(all_raw_text)
        cleaned_text: str = _clean_extracted_text(full_raw_text)
        
        print(f"   [Process {os.getpid()}] {filename} digital extraction finished. Extracted {len(cleaned_text.split())} words.")
        
        return filename, cleaned_text
            
    except Exception as e:
        print(f"\nFATAL ERROR: {filename} processing failed during digital extraction: {type(e).__name__} - {e}")
        return filename, None

def ocr_urdu_pdf(
    pdf_path: Path, 
    lang_code: str,
    start_page: int = 1, # 1-based index to start OCR (inclusive)
    end_page: Optional[int] = None # 1-based index to end OCR (inclusive)
) -> Tuple[str, Optional[str]]:
    """
    OCR worker using Tesseract for Urdu/Bengali PDFs. Processes one page at a time for 
    critical memory efficiency and is now constrained by page limits.
    """
    filename: str = pdf_path.name
    tesseract_langs: str = TESSERACT_LANG_MAP.get(lang_code, 'eng')
    print(f"   [Process {os.getpid()}] Starting TESSERACT OCR for {filename} (Lang: {tesseract_langs})...")
    
    if pytesseract is None or convert_from_path is None:
        print(f"   [Process {os.getpid()}] ERROR: OCR dependencies not installed.")
        return filename, None 
        
    try:
        # Get total number of pages efficiently using pypdf
        reader = PdfReader(str(pdf_path))
        num_pages = len(reader.pages)
        all_raw_text: List[str] = []
        
        # Calculate actual 1-based page range (needed for convert_from_path)
        actual_start = max(1, start_page)
        actual_end = num_pages if end_page is None else min(num_pages, end_page)

        if actual_start > actual_end:
            print(f"   [Process {os.getpid()}] WARNING: Invalid page range: Pages {actual_start}-{actual_end} are out of bounds or empty for {filename}.")
            return filename, None

        print(f"   [Process {os.getpid()}] Processing OCR pages {actual_start} to {actual_end}...")
        
        # Process one page at a time for low memory footprint (Crucial for low compute)
        for i in range(actual_start, actual_end + 1): # +1 because actual_end is inclusive 1-based
            
            # Use first_page/last_page to process only page i
            images = convert_from_path(
                str(pdf_path),
                dpi=400, # High DPI for accuracy
                fmt='png',
                first_page=i, # Use the current 1-based page number
                last_page=i,
                thread_count=1
            )
            
            if images:
                image = images[0]
                raw_text: str = pytesseract.image_to_string(
                    image,
                    lang=tesseract_langs,
                    config='--psm 3' # PSM 3 is usually best for a single column page
                )
                all_raw_text.append(raw_text.strip())
                del image # Explicitly delete the image object

            del images # Explicitly delete the list of images

        # Combine, cleanup, and return
        full_raw_text: str = "\n\n--- PAGE BREAK ---\n\n".join(all_raw_text)
        cleaned_text: str = re.sub(r'\n\s*\n', '\n\n', full_raw_text).strip()
        
        print(f"   [Process {os.getpid()}] {filename} OCR finished. Extracted {len(cleaned_text.split())} words.")
        return filename, cleaned_text
        
    except pytesseract.TesseractNotFoundError:
        print("\nFATAL ERROR: Tesseract executable not found. Please install Tesseract OCR.")
        return filename, None
    except Exception as e:
        print(f"\nFATAL ERROR: {filename} processing failed during OCR: {type(e).__name__} - {e}")
        return filename, None

def process_pdf_file(pdf_path: Path, lang_code: str) -> Tuple[str, Optional[str]]:
    """
    Selects the appropriate processing method based on the language folder, 
    applies special page constraints, and forces OCR for known-corrupted files.
    """
    filename = pdf_path.name
    
    # Check for special page constraints
    page_config = SPECIAL_PAGE_CONFIG.get(filename, {})
    start_page = page_config.get('start_page', 1)
    end_page = page_config.get('end_page', None)

    if page_config:
        print(f"   [Constraint Applied] Using pages {start_page} to {end_page} for {filename}.")
        
    # RULE: Force OCR for Urdu (corrupted text layer) AND this specific Bengali file (corrupted encoding).
    ocr_required = (lang_code == 'ur') or (lang_code== 'bn')

    if ocr_required:
        # Use the memory-efficient OCR worker (which handles 'ur' and 'bn' via lang_code map)
        return ocr_urdu_pdf(pdf_path, lang_code, start_page, end_page)
    else:
        # All other languages / files use fast digital extraction
        return extract_digital_pdf_text(pdf_path, lang_code, start_page, end_page)


def save_markdown_output(filename: str, lang_code: str, extracted_text: str) -> None:
    """Saves the extracted text to the required Markdown destination."""
    
    output_dir: Path = OUTPUT_BASE_DIR / lang_code
    output_dir.mkdir(parents=True, exist_ok=True)
    
    markdown_filename: str = filename.replace('.pdf', '.md')
    output_path: Path = output_dir / markdown_filename
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        print(f"  -> SUCCESS: Saved {markdown_filename} to {output_dir.relative_to(PROJECT_ROOT)}")
    except Exception as e:
        print(f"  -> ERROR: Failed to save file {output_path}: {e}")


# --- Main Execution Flow ---

def run_extraction_pipeline() -> None:
    """
    Orchestrates the entire parallel text extraction process based on the classification report.
    """
    if PdfReader is None:
        console.print("[red]CRITICAL DEPENDENCY WARNING [/red]")
        print("The pypdf library is required for digital PDF extraction. Please install it.")
        
    # Check Tesseract/OCR dependencies only if Urdu files are present, but check here for pre-flight warning
    if pytesseract is None or convert_from_path is None:
        if 'ur' in TESSERACT_LANG_MAP.keys(): # Simple check if we intend to use OCR
            console.print("[red]OCR DEPENDENCY WARNING[/red]")
            print("Tesseract OCR (pytesseract, pdf2image) is required for Urdu PDFs. Please install it.")
        
    console.print("🚀 [bold cyan]Starting Hybrid Multilingual PDF Extraction Pipeline for Digital PDFs[/bold cyan]")
    
    # 1. LOAD CLASSIFICATION REPORT
    report: Optional[Dict[str, Any]] = load_classification_report(CLASSIFICATION_REPORT_PATH)
    if not report or not report.get('results'):
        console.print("[red]Pipeline aborted: Cannot load or process classification report.[/red]")
        return
        
    # 2. GATHER ALL DIGITAL TASKS (All PDFs in the 'digital_pdfs' list)
    all_tasks: List[Tuple[Path, str]] = [] # List of (pdf_path, lang_code)
    
    print(" 🔍 Analyzing classification report for digital PDFs...")
    for lang_code, lang_data in report['results'].items():
        lang_folder: Path = PDF_BASE_DIR / lang_code
        
        # Check for files flagged as digital
        for filename in lang_data.get('digital_pdfs', []): 
            pdf_path: Path = lang_folder / filename
            if pdf_path.exists():
                all_tasks.append((pdf_path, lang_code))
            else:
                console.print(f"[red]WARNING[/red]: File listed in report not found: {pdf_path}")

    if not all_tasks:
        print("✅ No digital PDFs found to process. Pipeline finished.")
        return

    print(f"✅ Found {len(all_tasks)} digital PDFs to process. Starting parallel extraction.")

    # 3. PARALLEL EXECUTION
    
    MAX_WORKERS: int = os.cpu_count() or 4 
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        
        future_to_file: Dict[concurrent.futures.Future, Tuple[str, str]] = {}
        
        for pdf_path, lang_code in all_tasks:
            future = executor.submit(process_pdf_file, pdf_path, lang_code)
            future_to_file[future] = (pdf_path.name, lang_code)

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            filename, lang_code = future_to_file[future]
            try:
                # Result is (filename, extracted_text)
                _, extracted_text = future.result() 
                
                if extracted_text:
                    # 4. SAVE OUTPUT
                    save_markdown_output(filename, lang_code, extracted_text)
                else:
                    console.print(f"[red]WARNING[/red]: Extraction failed or returned empty for {filename}.")
                    
            except Exception as exc:
                console.print(f"[red]ERROR[/red]: {filename} generated an exception: {exc}")

    print("")
    console.print("🎉 [green]SUCCESS[/green]: Finished processing all specified digital PDFs.")

