#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path
import textwrap
from typing import List, Dict
from google.genai import Client

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR / "src"))

from llm.llm_query import API_KEY, MODEL_NAME

# ---------------------- Multilingual Prompt Tuning ----------------------

class MultilingualPrompt:
    """
    A helper class to create optimized prompts for multilingual queries
    to improve LLM responses in the RAG pipeline.
    """

    def __init__(self, user_query: str, retrieved_chunks: List[Dict], top_k: int = 5):
        """
        user_query: The query entered by user (any language)
        retrieved_chunks: List of dicts with keys ["text", "metadata", "distance"]
        top_k: Number of top chunks to consider in prompt
        """
        self.user_query = user_query.strip()
        self.retrieved_chunks = retrieved_chunks
        self.top_k = top_k

    def build_prompt(self) -> str:
        """
        Constructs a multilingual-aware prompt for the LLM using top chunks.
        """
        # Collect top context chunks
        all_contexts = []
        for chunk in sorted(self.retrieved_chunks, key=lambda x: x.get("distance", 1.0))[:self.top_k]:
            text = chunk.get("text", "").strip()
            lang = chunk.get("metadata", {}).get("language", "unknown")
            if text:
                all_contexts.append(f"[Language: {lang}] {text}")

        if not all_contexts:
            return "⚠️ No relevant context available to generate a prompt."

        combined_context = "\n\n".join(all_contexts)

        # Build a multilingual-aware prompt
        prompt = textwrap.dedent(f"""
        You are a multilingual AI assistant. Your task is to answer the user's question
        based solely on the context provided below. If the answer is not available in
        the context, explicitly say so. Provide the answer in the same language as the
        query, and also include an English translation if possible.

        Question: {self.user_query}

        Context:
        {combined_context}

        Guidelines:
        1. Answer concisely but clearly.
        2. Do not hallucinate; rely only on the context.
        3. Preserve the language formatting.
        4. Provide English translation below the original answer.

        Answer:
        """)
        return prompt

# ---------------------- LLM Call Wrapper ----------------------

def generate_multilingual_answer(user_query: str, retrieved_chunks: List[Dict], top_k: int = 5) -> str:
    """
    Calls Gemini-2.5-Flash with multilingual prompt tuning.
    """
    prompt_builder = MultilingualPrompt(user_query, retrieved_chunks, top_k)
    prompt = prompt_builder.build_prompt()

    if "⚠️" in prompt:
        return prompt  # No context available

    client = Client(api_key=API_KEY)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt]
        )

        if response and response.candidates:
            return response.candidates[0].content.parts[0].text.strip()
        else:
            return "⚠️ No valid response received from Gemini."
    except Exception as e:
        return f"❌ Error generating multilingual answer: {str(e)}"

# ---------------------- Usage Example ----------------------
if __name__ == "__main__":
    # Example retrieved chunks
    example_chunks = [
        {
            "text": "The individual had 9 family connections associated with SOL.",
            "metadata": {"language": "ur"},
            "distance": 0.12
        },
        {
            "text": "Responsibilities were assigned and recorded in BSE and LL systems.",
            "metadata": {"language": "ur"},
            "distance": 0.15
        }
    ]
    query = "Who had 9 family connections associated with SOL?"
    answer = generate_multilingual_answer(query, example_chunks)
    print("LLM Answer:\n", answer)
