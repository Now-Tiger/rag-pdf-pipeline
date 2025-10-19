#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Any

import fitz  # PyMuPDF
import os
import json
from datetime import datetime
from rich.console import Console
 
console = Console()


# --- Configuration for Robust Classification ---

# Number of pages to sample for classification (improves efficiency for large PDFs)
PAGE_SAMPLE_SIZE: int = 10 
# Average number of characters per sampled page required to be considered DIGITAL.
# This prevents misclassification due to spurious/junk text in image-based PDFs.
TEXT_THRESHOLD: int = 50 

# --- Core Classification Logic ---

def classify_pdf_type(file_path: str) -> str:
    """
    Classifies a PDF as 'scanned' (image-based) or 'digital' (text-based) 
    by checking the average text density across the first few pages.
    """
    total_text_chars = 0
    pages_checked = 0

    try:
        with fitz.open(file_path) as doc:
            
            # Check only a sample of pages for efficiency
            num_pages_to_check = min(doc.page_count, PAGE_SAMPLE_SIZE)
            
            for i in range(num_pages_to_check):
                page = doc.load_page(i)
                # Use 'text' output as it's the simplest measure of text presence
                text = page.get_text().strip()
                total_text_chars += len(text)
                pages_checked += 1
                
                # Optimization: If we find enough text early, classify immediately
                if total_text_chars > (TEXT_THRESHOLD * num_pages_to_check):
                    return "digital"
        if pages_checked == 0:
            # Handle empty documents or documents where sampling failed (shouldn't happen often)
            return "scanned" 

        # Calculate average text density
        avg_text_density = total_text_chars / pages_checked

        if avg_text_density >= TEXT_THRESHOLD:
            return "digital"
        else:
            return "scanned"

    except Exception as e:
        console.print(f"[red]Error processing {file_path}: {e}[/red]")
        # Default to scanned if processing fails, forcing OCR fallback if possible
        return "scanned"


def classify_pdfs_in_subfolders(base_folder: str, output_path: str = "classification_report_latest.json") -> Dict[str, Any]:
    """
    Classify PDFs as scanned or digital across multiple subfolders.
    Saves structured results to JSON for later use.
    """
    classification_results: Dict[str, Dict[str, List[str]]] = {}
    console.print(f"[bold cyan]Scanning base folder:[/bold cyan] {base_folder}\n")

    for root, dirs, files in os.walk(base_folder):
        # Only process leaf directories containing PDFs
        pdf_files = [f for f in files if f.lower().endswith(".pdf")]
        if not pdf_files:
            continue

        folder_name = os.path.basename(root)
        scanned: List[str] = []
        digital: List[str] = []

        for file in pdf_files:
            file_path = os.path.join(root, file)
            
            # Use the new, robust classification logic
            pdf_type = classify_pdf_type(file_path)
            
            if pdf_type == "scanned":
                scanned.append(file)
                status = "🖨️ Scanned"
                color = "red"
            else:
                digital.append(file)
                status = "📄 Digital"
                color = "green"
            
            console.print(f"[{color}]{status}[/{color}] — {folder_name}/{file}")

        classification_results[folder_name] = {
            "scanned_pdfs": scanned,
            "digital_pdfs": digital,
        }

    # Add metadata and save JSON
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "base_folder": base_folder,
        "page_sample_size": PAGE_SAMPLE_SIZE,
        "text_threshold": TEXT_THRESHOLD,
        "results": classification_results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    console.print(f"\n[bold yellow]✅ Classification complete! Results saved to:[/bold yellow] {output_path}")
    return output_data









































































































