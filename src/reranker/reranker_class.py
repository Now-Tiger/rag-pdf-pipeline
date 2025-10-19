#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

import torch
from sentence_transformers import CrossEncoder
from rich.console import Console

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Default reranker model — fast and CPU-friendly.
# You can switch to a stronger model (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2" or "vblagoje/bart_l2_cross-encoder")
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Device detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"[RERANKER] Using device: {DEVICE}")


class Reranker:
    """
    Reranker class using a cross-encoder model.

    Expected candidate format (list of dicts), matching retriever output:
      [
        {
          "id": "lang_doc_1",
          "document": "<chunk text>",
          "metadata": { ... },
          "distance": 0.123
        },
        ...
      ]

    The rerank(...) function returns the same objects with an added "rerank_score" field,
    sorted descending by rerank_score.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER, batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model: Optional[CrossEncoder] = None
        try:
            console.print(f"Loading reranker model: [bold]{model_name}[/bold] on {DEVICE} ...")
            self.model = CrossEncoder(model_name, device=DEVICE)
            console.print(f":white_check_mark: Loaded reranker: [green]{model_name}[/green]")
        except Exception as e:
            logger.warning(f"[RERANKER] Failed to load CrossEncoder {model_name}: {e}")
            self.model = None

    def _score_pairs(self, query: str, texts: List[str]) -> List[float]:
        """
        Score (query, text) pairs using the cross-encoder.
        Returns a list of float scores aligned with texts.
        """
        if self.model is None:
            raise RuntimeError("Reranker model not loaded")
        pairs = [(query, t) for t in texts]
        scores: List[float] = []
        # batch prediction to avoid OOM
        total = len(pairs)
        if total == 0:
            return []
        batch_size = self.batch_size
        for i in range(0, total, batch_size):
            batch = pairs[i : i + batch_size]
            try:
                batch_scores = self.model.predict(batch)  # returns list of floats
            except TypeError:
                # Some CrossEncoder versions expect a list of strings "query \t doc" — but predict(batch) should work.
                # As a fallback, try supplying as list of pairs flattened
                batch_scores = self.model.predict(batch)
            scores.extend([float(s) for s in batch_scores])
        return scores

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank a list of candidate dicts for a query.

        Parameters:
        - query: the user query string
        - candidates: list of dicts with keys "document", "id", "metadata", "distance" (optional)
        - top_k: if set, return only top_k reranked candidates

        Returns:
        - list of candidate dicts with an added key "rerank_score", sorted by that score (desc).
        """
        if not candidates:
            return []

        # Protect: if model not loaded, fallback to ordering by 'distance' if present (lower distance better)
        if self.model is None:
            logger.warning("[RERANKER] Model not available; falling back to original ordering (distance asc).")
            # Attach fallback score such that smaller distance -> higher score
            out = []
            for cand in candidates:
                dist = cand.get("distance", None)
                fallback_score = -dist if dist is not None else 0.0
                c = dict(cand)
                c["rerank_score"] = fallback_score
                out.append(c)
            out.sort(key=lambda x: x["rerank_score"], reverse=True)
            return out[:top_k] if top_k else out

        # Prepare texts to score
        texts = [c.get("document", "") for c in candidates]
        # score them in batches
        try:
            scores = self._score_pairs(query, texts)
        except Exception as e:
            logger.exception(f"[RERANKER] Scoring failed: {e}")
            # fallback to original ordering by distance
            return self._fallback_by_distance(candidates, top_k)

        # attach rerank score to candidates
        reranked = []
        for cand, score in zip(candidates, scores):
            c = dict(cand)
            c["rerank_score"] = float(score)
            reranked.append(c)

        # sort descending by rerank_score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        if top_k is not None:
            return reranked[:top_k]
        return reranked

    def _fallback_by_distance(self, candidates: List[Dict[str, Any]], top_k: Optional[int]) -> List[Dict[str, Any]]:
        out = []
        for cand in candidates:
            dist = cand.get("distance", None)
            fallback_score = -dist if dist is not None else 0.0
            c = dict(cand)
            c["rerank_score"] = fallback_score
            out.append(c)
        out.sort(key=lambda x: x["rerank_score"], reverse=True)
        return out[:top_k] if top_k else out


# -----------------------
# Example helper function
# -----------------------
# def rerank_and_print_example():
#     """
#     Example: how to call the reranker using the retriever output format.
#     """
#     # mock retrieval hits
#     hits = [
#         {"id": "ur_doc_1_1", "document": "قواعد مقرر ہيں ...", "metadata": {"source_pdf": "Extension-of-Ahdoc-Employees.pdf"}, "distance": 0.42},
#         {"id": "ur_doc_1_2", "document": "مزید تفصیلات ...", "metadata": {"source_pdf": "Extension-of-Ahdoc-Employees.pdf"}, "distance": 0.60},
#         {"id": "ur_doc_2_1", "document": "یہ اصول نافذ ہوتے ہیں ...", "metadata": {"source_pdf": "Notification-for-Other-Nationals.pdf"}, "distance": 0.71},
#     ]
#     query = "Rules regarding adhoc employees"
# 
#     r = Reranker()
#     reranked = r.rerank(query, hits, top_k=3)
# 
#     console.print("[bold green]Reranked results:[/bold green]")
#     for i, item in enumerate(reranked, start=1):
#         console.print(f"{i}. id={item['id']} score={item['rerank_score']:.4f} src={item['metadata'].get('source_pdf')}")
#         console.print(f"   {item['document'][:200]}...\n")
# 
# 
# # allow running module directly for a quick smoke-test
# if __name__ == "__main__":
#     rerank_and_print_example()
