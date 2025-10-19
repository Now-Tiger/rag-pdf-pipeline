#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict
import json
import logging
from rich.console import Console

import chromadb
from sentence_transformers import SentenceTransformer
from warnings import filterwarnings

# --- Project setup ---
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR / "src"))

from llm.llm_query import generate_llm_answer
from llm.multilingual_prompt import generate_multilingual_answer
from llm.chat_memory import add_interaction, get_conversation_history
# from llm.query_decomposer import QueryDecomposer

# ---------------------- LOGGING ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)
console = Console()

filterwarnings("always")
filterwarnings("ignore")

# ---------------------- CONFIG ----------------------
CHROMA_DIR = BASE_DIR / "data" / "vector_db" / "chroma"
COLLECTION_NAME = "rag_multilingual_docs"
TOP_K = 5
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# ---------------------- INIT ----------------------
console.print(f"🌍 Using multilingual embedding model: [green]{EMBEDDING_MODEL}[/green]")
model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


# ---------------------- FUNCTIONS ----------------------
def embed_query(query: str) -> List[float]:
    """Generate normalized embedding for the user query."""
    query_text = f"query: {query.strip()}"
    embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    return embedding.tolist()


def apply_metadata_filters(results: Dict, language: str = None, source_pdf: str = None) -> Dict:
    """Filter query results based on metadata."""
    filtered_results = {"documents": [], "metadatas": [], "distances": []}
    for doc, meta, dist in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0]
    ):
        if language and meta.get("language") != language:
            continue
        if source_pdf and meta.get("source_pdf") != source_pdf:
            continue

        filtered_results["documents"].append(doc)
        filtered_results["metadatas"].append(meta)
        filtered_results["distances"].append(dist)

    return filtered_results



def query_pipeline() -> Dict:
    """Full query pipeline: metadata filtering + retrieval + LLM answer with chat memory."""
    console.print("\n🖋️ Enter your query (any supported language):")
    query = input(">>> ").strip()
    if not query:
        console.print("[red]Query cannot be empty![/red]")
        return {}

    # Optional metadata filters
    console.print("\n🔹 Filter by language (e.g., 'ur', 'bn', 'zh') or press Enter to skip:")
    lang_filter = input(">>> ").strip() or None
    console.print("\n🔹 Filter by source PDF or press Enter to skip:")
    pdf_filter = input(">>> ").strip() or None

    try:
        # 🧠 Step 0: Display last few conversation turns (contextual memory)
        history = get_conversation_history(limit=3)
        if history:
            console.print("\n📜 [bold yellow]Recent Chat Memory:[/bold yellow]")
            for i, h in enumerate(history, start=1):
                console.print(f"[grey58]{i}. Q:[/grey58] {h['user_input']}")
                console.print(f"[grey58]   A:[/grey58] {h['llm_response']}\n")

        # Step 1: Embed query
        query_embedding = embed_query(query)

        # Step 2: Retrieve top chunks
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K
        )

        # Step 3: Apply metadata filters
        filtered = apply_metadata_filters(results, language=lang_filter, source_pdf=pdf_filter)

        if not filtered["documents"]:
            console.print("[yellow]⚠️ No similar content found for the given filters.[/yellow]")
            return {}

        # Step 4: Build reranked results
        reranked_results = []
        for idx, (doc, meta, dist) in enumerate(
            zip(filtered["documents"], filtered["metadatas"], filtered["distances"]), start=1
        ):
            reranked_results.append({
                "text": doc,
                "metadata": meta,
                "distance": dist,
            })

        # Step 5: Generate LLM answer
        console.print("\n🤖 [bold magenta]Generating LLM answer using Gemini-2.5-Flash...[/bold magenta]")
        try:
            llm_answer = generate_multilingual_answer(query, reranked_results)
            console.print(f"\n[bold green]LLM Answer:[/bold green]\n{llm_answer}")
        except Exception as e:
            logger.error(f"Error generating multilingual answer: {e}", exc_info=True)
            console.print(f"[red]❌ Error generating multilingual answer: {e}[/red]")
            console.print("\n[cyan]Falling back to default Gemini LLM pipeline...[/cyan]")
            llm_answer = generate_llm_answer(query, reranked_results)
            console.print(f"\n[bold green]Fallback LLM Answer:[/bold green]\n{llm_answer}")

        # 🧩 Step 6: Persist to chat memory
        try:
            add_interaction(query, llm_answer, reranked_results)
            console.print("[blue]💾 Interaction saved to chat memory.[/blue]")
        except Exception as e:
            console.print(f"[red]⚠️ Failed to save chat memory: {e}[/red]")

        return {"query": query, "results": reranked_results, "llm_answer": llm_answer}

    except Exception as e:
        logger.error(f"Error during query: {e}", exc_info=True)
        console.print(f"[red]Error during query: {e}[/red]")
        return {}



# ---------------------- MAIN LOOP ----------------------
if __name__ == "__main__":
    while True:
        try:
            result_data = query_pipeline()
            if result_data:
                with open("last_query_results.json", "w", encoding="utf-8") as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)
        except KeyboardInterrupt:
            console.print("\n[bold red]👋 Exiting CLI...[/bold red]")
            break

