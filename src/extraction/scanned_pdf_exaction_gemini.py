#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import tempfile
from dotenv import find_dotenv, load_dotenv

# Removed import glob
from pathlib import Path # <-- NEW: Import Path for robust path handling
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple, Dict, Any

# Third-party libraries
from pdf2image import convert_from_path
import pytesseract

# Using the official Google GenAI SDK
from google import genai
from google.genai.errors import APIError
from rich.console import Console

load_dotenv(find_dotenv())
console = Console()

# --- Configuration and Environment Variables ---
API_KEY: str = os.environ.get("GEMINI_API_KEY", "xxx")

if not API_KEY or API_KEY == "xxx":
    console.print(">>> [bold white]Enter your gemini api key - [/bold white] visit `https://aistudio.google.com/app/api-keys`")
    API_KEY = str(input(">>> "))

GEMINI_MODEL: str = "gemini-2.5-flash"

ROOT_DIR: Path = Path(__file__).resolve().parents[2] / 'data' / 'pdfs'
OUTPUT_FILENAME: str = "translated_document.md"

# IMPORTANT: Ensure Tesseract and Poppler are installed on your system.
# On Linux (Debian/Ubuntu):
# sudo apt-get install tesseract-ocr libpoppler-qt5-dev
# On Windows: Download and install the respective executables.
# You may need to specify the path to the Tesseract executable here:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Tesseract Languages: 'eng' for English, 'urd' for Urdu.
TESSERACT_LANGUAGES: str = 'eng+urd' 

# Initialize the GenAI client globally
# The client will automatically pick up the GEMINI_API_KEY environment variable if API_KEY is empty.
client: Optional[genai.Client] = None
try:
    client = genai.Client(api_key=API_KEY if API_KEY else None)
except Exception as e:
    print(f"ERROR initializing Google GenAI client. Ensure your API Key is set correctly: {e}")


# --- Utility Functions ---

def find_single_pdf(root_dir: Path) -> Optional[str]:
    """
    Searches the specified root directory for a single PDF file 
    at the pattern: {root_dir}/*.pdf
    
    The function now accepts a pathlib.Path object.
    """
    try:
        # Use Path.glob() for a robust search
        pdf_files: List[Path] = list(root_dir.glob("*.pdf")) 
        
        root_dir_str = str(root_dir)

        if not pdf_files:
            print(f"ERROR: No PDF file found in the expected path '{root_dir_str}/*.pdf'.") 
            return None
        if len(pdf_files) > 1:
            # pdf_files[0] is a Path object, which prints cleanly
            print(f"WARNING: Found {len(pdf_files)} PDFs. Using the first one: {pdf_files[0]}")
        
        # Return the string representation of the path for compatibility with convert_from_path
        return str(pdf_files[0])
    except Exception as e:
        print(f"An error occurred during file search: {e}")
        return None



def preprocess_and_ocr(page_number: int, pdf_path: str, temp_dir: str) -> Tuple[int, Optional[str]]:
    """
    Performs PDF page to image conversion and OCR for a single page.
    This function is designed to run in parallel processes.
    
    Returns: A tuple of (page_number, extracted_text).
    """
    try:
        print(f"   [Process {os.getpid()}] Starting page {page_number + 1}...")
        
        # 1. PDF Page to Image Conversion (High DPI for better OCR)
        images: List[Any] = convert_from_path(
            pdf_path, 
            dpi=400, # High DPI is critical for scanned documents
            first_page=page_number + 1, 
            last_page=page_number + 1,
            fmt='png',
            thread_count=1 # Keep threading internal to pdf2image minimal, rely on ProcessPoolExecutor
        )
        
        if not images:
            return page_number, None

        img_path: str = os.path.join(temp_dir, f"page_{page_number + 1}.png")
        # Save the image to the temporary directory
        images[0].save(img_path, 'PNG')

        # 2. OCR Execution
        raw_text: str = pytesseract.image_to_string(
            images[0], 
            lang=TESSERACT_LANGUAGES,
            config='--psm 3' # PSM 3 is usually best for a single column page
        )
        
        # Simple cleanup of common OCR errors (e.g., excessive newlines)
        cleaned_text: str = re.sub(r'\n\s*\n', '\n\n', raw_text.strip())
        print(f"   [Process {os.getpid()}] Page {page_number + 1} OCR finished. Extracted {len(cleaned_text.split())} words.")
        
        return page_number, cleaned_text
        
    except pytesseract.TesseractNotFoundError:
        print("\nFATAL ERROR: Tesseract executable not found. Please install Tesseract OCR.")
        return page_number, None
    except Exception as e:
        print(f"\nFATAL ERROR: Page {page_number + 1} processing failed: {e}")
        return page_number, None




def translate_and_synthesize_gemini(raw_text_block: str) -> str:
    """
    Calls the Gemini API using the google-genai client to translate and synthesize the text block.
    """
    global client
    if client is None or not raw_text_block:
        return ""
        
    print(f"  -> Sending block of {len(raw_text_block.split())} words to Gemini for translation.")

    # System instruction: Highly optimized for the desired translation/synthesis task.
    system_prompt: str = (
        "You are a professional translator and document synthesizer. "
        "Analyze the provided text block, which contains mixed Urdu and English content extracted via OCR. "
        "Translate all Urdu text into accurate, clear, and formal English. "
        "Preserve all original English content as is. "
        "Combine the translated and original text into a coherent, clean Markdown document format. "
        "**CRITICAL**: Do not include any commentary, title, preamble, or wrapper text outside of the synthesized English document content itself."
    )
    
    try:
        # Using the official client method
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=raw_text_block, # The client handles the parts/text structure automatically
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                # Setting a low temperature for predictable translation results
                temperature=0.1 
            )
        )
        
        translated_text = response.text
        
        print("  <- Translation complete.")
        return translated_text
        
    except APIError as e:
        print(f"  !!! GenAI API Error: {e}")
        return "[[TRANSLATION ERROR: GenAI API Error]]\n"
    except Exception as e:
        print(f"  !!! An unexpected error occurred during GenAI call: {e}")
        return "[[TRANSLATION ERROR: Unexpected Error]]\n"


def run_mlops_pipeline() -> None:
    """
    Orchestrates the entire parallel OCR and sequential translation process.
    """
    global client
    print("--- 🚀 Starting High-Performance PDF Translation Pipeline ---")

    if client is None:
        print("Pipeline aborted: GenAI client is not initialized.")
        return
    
    # 1. FIND THE PDF
    pdf_path: Optional[str] = find_single_pdf(ROOT_DIR)
    if not pdf_path:
        print("Pipeline aborted.")
        return

    print(f"✅ Found PDF: {pdf_path}")
    
    # Use a temporary directory for image files, ensuring cleanup
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Determine the number of pages for parallelism
        try:
            # Get info about the PDF to determine page count
            # This is a fast operation
            pages = convert_from_path(pdf_path, first_page=1, last_page=1)
            # A dummy call to get the total number of pages
            pages = convert_from_path(pdf_path, last_page=99999) 
            num_pages: int = len(pages)
            pages = None # Release memory
        except Exception as e:
            print(f"Error determining PDF page count: {e}")
            return
            
        print(f"📄 PDF has {num_pages} pages. Utilizing all available cores.")

        # 2. PARALLEL OCR EXECUTION (CPU-BOUND)
        ocr_results: Dict[int, str] = {}
        # Maximize parallelism for the OCR/Image conversion
        MAX_WORKERS: int = os.cpu_count() or 4 

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all pages to the process pool
            future_to_page: Dict[Any, int] = {
                executor.submit(preprocess_and_ocr, i, pdf_path, temp_dir): i 
                for i in range(num_pages)
            }
            
            # Process results as they complete (as_completed for max throughput)
            for future in as_completed(future_to_page):
                page_num, text = future.result()
                if text:
                    ocr_results[page_num] = text

        print(f"\n✅ All pages processed by OCR. Successfully extracted text from {len(ocr_results)} pages.")

        # 3. SEQUENCE AND BATCH TEXT BLOCKS
        # Sort results by page number
        sorted_blocks: List[str] = [ocr_results[i] for i in sorted(ocr_results.keys())]
        
        # Combine all OCR text into one large block for a single, comprehensive LLM call
        # This is often faster and better for context than page-by-page translation.
        full_raw_text: str = "\n\n-- PAGE BREAK --\n\n".join(sorted_blocks)

        if not full_raw_text:
            print("❌ Extracted text was empty. Check Tesseract installation and PDF quality.")
            return

        # 4. GEMINI TRANSLATION AND SYNTHESIS (I/O-BOUND)
        print("\n🧠 Starting LLM translation and synthesis...")

        final_english_markdown: str = translate_and_synthesize_gemini(full_raw_text)

        # 5. FINAL STORAGE
        if final_english_markdown:
            with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
                f.write(final_english_markdown)
                
            print(f"\n🎉 SUCCESS: Document translated and saved to {OUTPUT_FILENAME}")
        else:
            print("\n❌ FAILED: LLM returned empty or encountered an unrecoverable error.")


if __name__ == '__main__':
    run_mlops_pipeline()

