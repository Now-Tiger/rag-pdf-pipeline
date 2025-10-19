
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer
from rich.console import Console

# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("data/logs/embeddings.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
console = Console()

# ------------------------------------------------------------
# Config paths
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_MD_DIR = BASE_DIR / "data" / "processed" / "extracted_md"
CHROMA_DIR = BASE_DIR / "data" / "vector_db" / "chroma"
CLASSIFICATION_REPORT = BASE_DIR / "data" / "classification_report.json"

# ------------------------------------------------------------
# Initialize model and Chroma
# ------------------------------------------------------------
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"  # multilingual model
model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

CHROMA_DIR.mkdir(parents=True, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# ✅ Single unified multilingual collection
COLLECTION_NAME = "rag_multilingual_docs"
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks to avoid OOM issues."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def read_markdown_files(language_dir: Path) -> Dict[str, List[str]]:
    """Read and chunk markdown files into smaller pieces per document."""
    data = {}
    for md_file in language_dir.glob("*.md"):
        try:
            text = md_file.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                chunks = chunk_text(text)
                data[md_file.stem] = chunks
        except Exception as e:
            logger.error(f"❌ Failed to read {md_file.name}: {e}")
    return data


def embed_texts(texts: List[str], batch_size: int = 8) -> List[List[float]]:
    """Efficient batched embedding with normalization."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def store_in_chroma(language: str, texts_map: Dict[str, List[str]]) -> None:
    """Store multilingual embeddings into a single Chroma collection."""
    logger.info(f"🌍 Processing language '{language}' ...")

    all_texts, ids, metadatas = [], [], []
    for doc_name, chunks in texts_map.items():
        for idx, chunk in enumerate(chunks, start=1):
            ids.append(f"{language}_{doc_name}_{idx}")
            all_texts.append(chunk)
            metadatas.append(
                {
                    "language": language,
                    "source_pdf": f"{doc_name}.pdf",
                    "chunk_index": idx,
                }
            )

    if not all_texts:
        logger.warning(f"⚠️ No text chunks found for '{language}'")
        return

    embeddings = embed_texts(all_texts)
    collection.add(ids=ids, embeddings=embeddings, documents=all_texts, metadatas=metadatas)

    logger.info(f"✅ Stored {len(all_texts)} chunks for '{language}' in '{COLLECTION_NAME}'.")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def generate_and_store_embeddings() -> None:
    console.print("🚀 [bold cyan]Starting multilingual embeddings generation...[/bold cyan]")

    if not PROCESSED_MD_DIR.exists():
        raise FileNotFoundError(f"Processed markdown directory not found: {PROCESSED_MD_DIR}")

    with open(CLASSIFICATION_REPORT, "r", encoding="utf-8") as f:
        report = json.load(f)

    results = report.get("results", {})
    if not results:
        raise ValueError("No results found in classification_report.json")

    for lang in results.keys():
        lang_dir = PROCESSED_MD_DIR / lang
        if not lang_dir.exists():
            logger.warning(f"⚠️ Language folder missing: {lang_dir}")
            continue

        texts_map = read_markdown_files(lang_dir)
        if not texts_map:
            logger.warning(f"⚠️ No markdown data found for language: {lang}")
            continue

        store_in_chroma(lang, texts_map)

    console.print("🏁 [green]Embeddings generation + Chroma persistence completed successfully![/green]")


# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------
if __name__ == "__main__":
    generate_and_store_embeddings()



