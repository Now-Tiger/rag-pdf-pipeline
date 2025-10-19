#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from rich.console import Console
import json

from llm.llm_query import generate_llm_answer

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR / "src"))
from llm.llm_query import generate_llm_answer


console = Console()


class QueryDecomposer:
    """
    Breaks a complex/multi-part query into simpler sub-queries
    for better retrieval in the RAG pipeline.
    """

    def __init__(self, max_subqueries: int = 3):
        """
        Args:
            max_subqueries: Maximum number of sub-queries to generate
        """
        self.max_subqueries = max_subqueries

    def decompose(self, query: str) -> List[str]:
        """
        Generate simpler sub-queries using the LLM.

        Args:
            query: The original complex query string

        Returns:
            List[str]: List of decomposed sub-queries
        """
        if not query.strip():
            console.print("[red]Query cannot be empty for decomposition[/red]")
            return []

        # Construct the prompt as a single query
        prompt = (
            f"Break the following query into up to {self.max_subqueries} "
            f"simpler, independent sub-queries, returning them as a JSON list of strings.\n\n"
            f'Query: "{query}"'
        )

        try:
            # Call generate_llm_answer with empty reranked_results
            llm_response = generate_llm_answer(prompt, reranked_results=[])

            # Attempt to parse as JSON first
            try:
                subqueries = json.loads(llm_response)
                if isinstance(subqueries, list):
                    return [sq.strip() for sq in subqueries if sq.strip()]
            except json.JSONDecodeError:
                # fallback: split by lines or bullets
                lines = [line.strip("-•* \n") for line in llm_response.splitlines()]
                return [line for line in lines if line]

        except Exception as e:
            console.print(f"[red]Error decomposing query: {e}[/red]")
            return []

        return []
