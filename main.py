#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from src.ingestion.classifier import classify_pdfs_in_subfolders
from src.extraction import scanned_multilingual_pdf_extractor, digital_multilingual_pdf_extractor
from src.chunking import chunk_texts
from src.embeddings import generate_embeddings


def main():
    base_folder: str = "data/pdfs"
    classify_pdfs_in_subfolders(base_folder, output_path="data/classification_report.json")

    scanned_multilingual_pdf_extractor.run_extraction_pipeline()
    print("=" * 100)
    digital_multilingual_pdf_extractor.run_extraction_pipeline()
    print("=" * 100)
    chunk_texts.proceed()
    print("=" * 100)
    generate_embeddings.generate_and_store_embeddings()


if __name__ == "__main__":
    main()
