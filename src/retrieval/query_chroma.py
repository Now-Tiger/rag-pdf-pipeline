#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer
from rich.console import Console

console = Console()

# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Config paths
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
CHROMA_DIR = BASE_DIR / "data" / "vector_db" / "chroma"

# ------------------------------------------------------------
# Initialize embedding model
# ------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
model = SentenceTransformer(EMBEDDING_MODEL)

# ------------------------------------------------------------
# Initialize Chroma Persistent Client
# ------------------------------------------------------------
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def embed_query(query: str) -> List[float]:
    """Convert user query into embedding vector."""
    return model.encode([query], convert_to_numpy=True).tolist()[0]


def semantic_search(query: str, language: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Query ChromaDB semantic search.
    Returns list of top-k matching chunks with metadata.
    """
    logger.info(f"Performing semantic search for language='{language}' ...")

    collection_name = language
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        logger.error(f"Collection '{collection_name}' not found: {e}")
        return []

    query_embedding = embed_query(query)

    # Query ChromaDB for top-k matches
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    # Collect results
    matches = []
    for idx in range(len(results["ids"][0])):
        matches.append(
            {
                "id": results["ids"][0][idx],
                "document": results["documents"][0][idx],
                "metadata": results["metadatas"][0][idx],
                "distance": results["distances"][0][idx],
            }
        )

    logger.info(f"Found {len(matches)} results for query='{query}'")
    return matches


# ------------------------------------------------------------
# CLI / Test
# ------------------------------------------------------------
if __name__ == "__main__":
    test_query = "What are the rules for extension of adhoc employees?"
    test_language = "ur"  # Language code based on classification

    top_matches = semantic_search(test_query, test_language, top_k=5)
    for i, m in enumerate(top_matches, 1):
        print(f"\nMatch {i}:")
        print(f"Source PDF: {m['metadata']['source_pdf']}")
        print(f"Chunk index: {m['metadata']['chunk_index']}")
        print(f"Distance: {m['distance']:.4f}")
        print(f"Content:\n{m['document'][:500]}...")  # print first 500 chars
