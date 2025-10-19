#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import logging
import textwrap

from google.genai import Client, types
from dotenv import load_dotenv, find_dotenv
from rich.console import Console


logger = logging.getLogger(__name__)
load_dotenv(find_dotenv(".env"))
console = Console()


API_KEY = os.environ.get("GEMINI_API_KEY", "xxx")

if not API_KEY or API_KEY == "xxx":
    console.print(">>> [bold white]Enter your gemini api key - [/bold white] visit `https://aistudio.google.com/app/api-keys`")
    API_KEY = str(input(">>> "))

MODEL_NAME = "gemini-2.5-flash"


def generate_llm_answer(query: str, reranked_results: list, top_k: int = 5) -> str:
    """
    Use Gemini-2.5-Flash to generate a final natural language answer
    based on the retrieved and reranked chunks.

    Parameters:
    - query: user query string
    - reranked_results: list of dicts with keys 'text', 'score', 'metadata', 'distance'
    - top_k: number of top chunks to include in context
    """
    # 🧠 Collect top chunks across all items
    all_contexts = []
    # Sort reranked_results by score descending
    sorted_items = sorted(
        reranked_results, key=lambda x: x.get("score", 0), reverse=True
    )[:top_k]
    for item in sorted_items:
        text = item.get("text", "")
        lang = item.get("metadata", {}).get("language", "unknown")
        if text:
            all_contexts.append(f"[Language: {lang}] {text.strip()}")

    if not all_contexts:
        return "⚠️ Sorry, I couldn’t find enough relevant context to answer that."

    # Combine top chunks into a single context block
    combined_context = "\n\n".join(all_contexts)

    # Construct multilingual-aware prompt for Gemini
    prompt = textwrap.dedent(
        f"""
    You are a multilingual AI assistant. 
    Your job is to answer the user's question based only on the given context below. 
    If you are not confident, explicitly say that the answer is not available in the provided context.

    Question: {query}

    Context:
    {combined_context}

    Provide your answer in the same language as the question and below that provide a translated version in english.
    """
    )

    # ⚡ Initialize Gemini client
    client = Client(api_key=API_KEY)

    # 🔮 Call Gemini model
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt],
        )

        # Step 4: Extract response safely
        if response and getattr(response, "candidates", None):
            candidate = response.candidates[0]
            if candidate and getattr(candidate, "content", None):
                return candidate.content.parts[0].text.strip()
        return "⚠️ Gemini could not generate an answer for this query."

    except Exception as e:
        return f"❌ Error generating answer from Gemini: {str(e)}"
