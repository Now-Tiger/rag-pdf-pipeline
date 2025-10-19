#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Tuple

import chromadb
from langdetect import detect
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

import logging
from rich.console import Console
from warnings import filterwarnings

BASE_DIR = Path(__file__).resolve().parents[2]  # go 2 levels up
sys.path.append(str(BASE_DIR / "src"))

from reranker.reranker_class import Reranker


filterwarnings("always")
filterwarnings("ignore")

# ---------------------- SETUP LOGGING ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
console = Console()

# ---------------------- CONSTANTS ----------------------
CHROMA_DB_DIR = "data/vector_db/chroma"
SUPPORTED_LANGS = ["ur", "bn", "zh", "en"]  # Urdu, Bengali, Chinese, English

# ---------------------- DEVICE CONFIG ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ---------------------- EMBEDDING MODEL ----------------------
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"  # multilingual model
embedder = SentenceTransformer(EMBEDDING_MODEL, device=str(device))

# ---------------------- Initialize reranker globally ----------------------
RERANKER = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size=16)

# ---------------------- CHROMA CLIENT ----------------------
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

# ---------------------- TRANSLATION MODEL (Unified) ----------------------
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL).to(device)

# NLLB language code mapping
NLLB_LANG_CODES = {
    "en": "eng_Latn",
    "ur": "urd_Arab",
    "bn": "ben_Beng",
    "zh": "zho_Hans",
}


def translate_query(text: str, target_lang: str) -> str:
    """
    Translate text using the NLLB-200 model.
    target_lang: short code ('ur', 'bn', 'zh', 'en')
    """
    try:
        if target_lang not in NLLB_LANG_CODES:
            raise ValueError(f"Unsupported target language: {target_lang}")
        target_code = NLLB_LANG_CODES[target_lang]

        # ensure the model is on the same device
        model.to(device)

        inputs = tokenizer(text, return_tensors="pt").to(device)
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_code],
            max_length=512,
        )
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        logger.info(f"Translated query → {target_lang}: {translated_text}")
        return translated_text
    except Exception as e:
        logger.warning(f"Translation failed for {target_lang}: {e}")
        return text


# ---------------------- CORE FUNCTIONS ----------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    """Compute embeddings using BGE-M3 model."""
    return embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()


def detect_query_language(query: str) -> str:
    """Detect query language."""
    try:
        lang = detect(query)
        logger.info(f"Detected query language: {lang}")
        return lang if lang in SUPPORTED_LANGS else "en"
    except Exception:
        return "en"


def get_or_create_collection(name: str):
    """Return an existing Chroma collection or create one if not found."""
    try:
        return chroma_client.get_collection(name)
    except Exception:
        logger.warning(f"Collection '{name}' not found. Creating a new one...")
        return chroma_client.create_collection(name)


def retrieve_similar_chunks(query: str, top_k: int = 5) -> dict:
    """
    Retrieve top-k relevant markdown chunks from multilingual Chroma DB.
    Always returns a dictionary: {lang: [(text, score, metadata), ...]}
    """
    query_lang = detect_query_language(query)
    results = {}

    for lang in SUPPORTED_LANGS:
        collection_name = "rag_multilingual_docs"
        try:
            collection = chroma_client.get_collection(collection_name)
        except Exception:
            logger.warning(f"No collection found for {lang}, skipping...")
            results[lang] = []
            continue

        search_query = query
        if query_lang == "en" and lang != "en":
            search_query = translate_query(query, lang)

        query_embedding = embed_texts([search_query])[0]
        res = collection.query(query_embeddings=[query_embedding], n_results=top_k)

        # -------------------- Normalize --------------------
        documents_raw = res.get("documents", [[]])
        distances_raw = res.get("distances", [[]])
        metadatas_raw = res.get("metadatas", [[]])

        # Flatten lists safely
        documents = documents_raw[0] if isinstance(documents_raw[0], list) else [documents_raw[0]]
        distances = distances_raw[0] if isinstance(distances_raw[0], list) else [distances_raw[0]]
        metadatas = metadatas_raw[0] if isinstance(metadatas_raw[0], list) else [metadatas_raw[0]]

        # Ensure same length
        min_len = min(len(documents), len(distances), len(metadatas))
        documents = documents[:min_len]
        distances = distances[:min_len]
        metadatas = metadatas[:min_len]

        # Final results: list of tuples (text, score, metadata)
        results[lang] = list(zip(documents, distances, metadatas))

    return results


# ---------------------- TEST BLOCK ----------------------
if __name__ == "__main__":
    sample_query = "দেশের সার্বিক উন্নয়ন-সংশ্রিষ্ট সমসাময়িক ও জনগুরুত্রপূর্ণ বিষয়ের উপর গবেষণা কার্যক্রম"
    hits = retrieve_similar_chunks(sample_query, top_k=3)

    for lang, docs in hits.items():
        console.print(f"[bold cyan]\nTop results for language [{lang}][/bold cyan]:")
        for text, score in docs:
            console.print(f"  → {text[:100]}... [green](score: {score:.4f})[/green]")
