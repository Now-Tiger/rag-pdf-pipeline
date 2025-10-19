#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import json
import os
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console

# Optional accurate tokenizer (preferred). Fallback to whitespace tokenizer.
try:
    import tiktoken  # type: ignore
    _TIKTOKEN_AVAILABLE = True
except Exception:
    _TIKTOKEN_AVAILABLE = False


console = Console()

# =========================
# Project paths (dynamic)
# =========================
ROOT_DIR = Path(__file__).resolve().parents[2]  # rag-pdf-pipeline/
DATA_DIR = ROOT_DIR / "data"
EXTRACTED_MD_DIR = DATA_DIR / "processed" / "extracted_md"
OUTPUT_CHUNKS_PATH = DATA_DIR / "processed" / "chunks.jsonl"
CHUNKING_REPORT_PATH = DATA_DIR / "processed" / "chunking_report.json"
LOG_PATH = DATA_DIR / "logs" / "chunking.log"

# =========================
# Config (tunable)
# =========================
# Target token budget per chunk (for embeddings / retrieval)
MAX_TOKENS = 250  # typical compact embedding chunk; tune for your embedding model/context
OVERLAP_TOKENS = 50
MIN_TOKENS = 32  # small guardrail to avoid tiny chunks

# Parallelism
MAX_WORKERS = min(8, (os.cpu_count() or 2))  # don't overload NFS/IO in small machines

# Tiktoken model name to approximate tokenization if available (optional)
TIKTOKEN_MODEL = "cl100k_base"  # good generic tokenizer; adjust to your embeddings model if desired

# =========================
# Utilities
# =========================
def _log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{ts} | {msg}\n")

def deterministic_chunk_id(source: str, start_char: int, end_char: int) -> str:
    key = f"{source}::{start_char}:{end_char}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

# -------------------------
# Tokenization utilities
# -------------------------
def _init_tokenizer():
    if _TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.get_encoding(TIKTOKEN_MODEL)  # type: ignore
        except Exception:
            # fallback to cl100k_base if specific model not present
            enc = tiktoken.get_encoding("cl100k_base")  # type: ignore
        return enc
    return None

_TOKENIZER = _init_tokenizer()

def count_tokens(text: str) -> int:
    if _TOKENIZER is not None:
        return len(_TOKENIZER.encode(text))
    # fallback: approximate by whitespace + punctuation heuristic
    # This is conservative: assume ~1.3 words/token for unknown tokenizers
    words = re.findall(r"\S+", text)
    approx = int(len(words) / 1.0)  # keep as words count (fast, deterministic)
    return max(1, approx)

def tokenize_to_words(text: str) -> List[str]:
    # simple split preserving words - used as a fallback to slice approx token counts
    words = re.findall(r"\S+", text)
    return words

# =========================
# Text cleaning & splitting
# =========================
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。\.!?؟۔\u0964\u0965])\s+")  # unicode-aware sentence boundary heuristic
# Note: includes punctuation used in some Indic & Arabic scripts. Not perfect but fast.

def normalize_text(text: str) -> str:
    # Trim BOMs, normalize windows newlines, remove excessive spaces
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # collapse multiple whitespace into single space while preserving paragraph breaks
    # preserve double-newlines for paragraph separation
    text = re.sub(r"\n{3,}", "\n\n", text)
    # strip leading/trailing whitespace
    text = text.strip()
    return text

def split_into_sentences(text: str) -> List[str]:
    # Fast heuristic sentence splitter:
    parts = _SENTENCE_SPLIT_RE.split(text)
    # Further split long lines by newline if necessary
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if "\n" in p and len(p) > 500:
            # break long newline blocks into smaller paragraphs
            sub = [s.strip() for s in p.split("\n") if s.strip()]
            out.extend(sub)
        else:
            out.append(p)
    return out

# =========================
# Chunking Algorithm
# =========================
def chunk_text_heuristic(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP_TOKENS) -> List[Dict[str, Any]]:
    """
    Create chunks by grouping sentences, trying to get near `max_tokens` per chunk,
    and ensuring `overlap` tokens between adjacent chunks.
    Returns a list of dicts: {text, start_char, end_char, token_count}
    """
    text = normalize_text(text)
    if not text:
        return []

    sentences = split_into_sentences(text)
    # Build running window of sentences to hit the token budget
    chunks = []
    cur_sentences: List[str] = []
    cur_char_start = 0  # will compute from text.find of first sentence in window later
    char_cursor = 0
    # We'll also preserve original char positions by walking through text
    sentence_positions: List[Tuple[str, int, int]] = []  # (sentence, start_char, end_char)
    search_pos = 0
    for s in sentences:
        # find s in text from current search_pos
        idx = text.find(s, search_pos)
        if idx == -1:
            # fallback: use previous search_pos
            idx = search_pos
        start = idx
        end = idx + len(s)
        sentence_positions.append((s, start, end))
        search_pos = end

    i = 0
    n = len(sentence_positions)
    while i < n:
        # build chunk starting at sentence i
        cur_tokens = 0
        start_char = sentence_positions[i][1]
        j = i
        last_end = sentence_positions[i][2]
        collected_texts = []
        while j < n:
            s, s_start, s_end = sentence_positions[j]
            # compute tentative token count if we include this sentence
            tentative_text = " ".join(collected_texts + [s])
            tentative_tokens = count_tokens(tentative_text)
            # if tentative exceeds max_tokens but current chunk is empty, allow it (long sentence)
            if tentative_tokens > max_tokens and collected_texts:
                break
            collected_texts.append(s)
            last_end = s_end
            cur_tokens = tentative_tokens
            j += 1
        chunk_text = text[start_char:last_end].strip()
        if not chunk_text:
            i = j or (i + 1)
            continue
        chunk = {
            "text": chunk_text,
            "start_char": start_char,
            "end_char": last_end,
            "token_count": cur_tokens,
            "char_count": len(chunk_text),
        }
        chunks.append(chunk)

        # move i forward with overlap handling
        if overlap <= 0:
            i = j
        else:
            # find the sentence index k such that overlap tokens approx are preserved
            # walk backward from j-1 to find earliest k where tokens of sentences[k..j-1] >= overlap
            if j <= i:
                i = j + 1
            else:
                # compute tokens window backwards
                k = j - 1
                overlap_accum = 0
                while k >= i:
                    # tokens in sentences[k..j-1]
                    segment_text = " ".join([sentence_positions[t][0] for t in range(k, j)])
                    overlap_accum = count_tokens(segment_text)
                    if overlap_accum >= overlap:
                        break
                    k -= 1
                # if we couldn't reach overlap, step by 1 to avoid infinite loop
                if k <= i:
                    i = i + 1
                else:
                    i = k

    # post-process: merge too small chunks with neighbors
    final_chunks = []
    idx = 0
    while idx < len(chunks):
        c = chunks[idx]
        if c["token_count"] < MIN_TOKENS and idx + 1 < len(chunks):
            # merge with next
            next_c = chunks[idx + 1]
            merged = {
                "text": (c["text"] + "\n" + next_c["text"]).strip(),
                "start_char": c["start_char"],
                "end_char": next_c["end_char"],
                "token_count": count_tokens(c["text"] + " " + next_c["text"]),
                "char_count": c["char_count"] + next_c["char_count"],
            }
            final_chunks.append(merged)
            idx += 2
        else:
            final_chunks.append(c)
            idx += 1

    return final_chunks

# =========================
# File processing
# =========================
def process_single_md(md_path: Path) -> List[Dict[str, Any]]:
    """
    Read markdown, chunk it, and produce list of chunk dicts with metadata.
    """
    try:
        raw = md_path.read_text(encoding="utf-8")
    except Exception:
        raw = md_path.read_text(encoding="latin-1")

    text = normalize_text(raw)
    if not text:
        return []

    chunks = chunk_text_heuristic(text)

    prepared_chunks: List[Dict[str, Any]] = []
    for idx, c in enumerate(chunks):
        start = c["start_char"]
        end = c["end_char"]
        chunk_text = c["text"]
        token_count = c["token_count"]
        char_count = c["char_count"]
        chunk_id = deterministic_chunk_id(str(md_path.resolve()), start, end)
        # derive metadata
        # folder code = parent folder name under extracted_md
        try:
            folder_code = md_path.parent.name
        except Exception:
            folder_code = ""
        prepared = {
            "id": chunk_id,
            "source_path": str(md_path.resolve()),
            "source_doc": md_path.name,
            "folder": folder_code,
            "chunk_index": idx,
            "text": chunk_text,
            "token_count": token_count,
            "char_count": char_count,
            "start_char": start,
            "end_char": end,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        prepared_chunks.append(prepared)
    return prepared_chunks

# =========================
# Orchestration
# =========================
def discover_md_files(base_dir: Path = EXTRACTED_MD_DIR) -> List[Path]:
    if not base_dir.exists():
        raise FileNotFoundError(f"Extracted md dir not found: {base_dir}")
    md_files = list(base_dir.rglob("*.md"))
    # deterministic ordering for reproducibility
    md_files.sort(key=lambda p: (str(p.parent), str(p.name)))
    return md_files

def run_chunking(worker_count: int = MAX_WORKERS):
    md_paths = discover_md_files()
    total_files = len(md_paths)
    _log(f"CHUNKING START: files={total_files}, workers={worker_count}, max_tokens={MAX_TOKENS}, overlap={OVERLAP_TOKENS}")
    chunk_count = 0
    docs_processed = 0
    # write JSONL streaming
    OUTPUT_CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # wipe previous outputs
    if OUTPUT_CHUNKS_PATH.exists():
        OUTPUT_CHUNKS_PATH.unlink()

    with ThreadPoolExecutor(max_workers=worker_count) as ex, open(OUTPUT_CHUNKS_PATH, "a", encoding="utf-8") as out_f:
        futures = {ex.submit(process_single_md, md): md for md in md_paths}
        for fut in as_completed(futures):
            md = futures[fut]
            try:
                chunks = fut.result()
                docs_processed += 1
                for ch in chunks:
                    # write compact JSON line
                    out_f.write(json.dumps(ch, ensure_ascii=False) + "\n")
                    chunk_count += 1
                _log(f"CHUNKED: {md} -> chunks={len(chunks)}")
            except Exception as e:
                _log(f"ERROR chunking {md}: {e}")

    # write a compact report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_md_files": total_files,
        "chunks_generated": chunk_count,
        "docs_processed": docs_processed,
        "max_tokens": MAX_TOKENS,
        "overlap_tokens": OVERLAP_TOKENS,
        "tiktoken_available": _TIKTOKEN_AVAILABLE,
    }
    CHUNKING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKING_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"CHUNKING COMPLETE: docs={docs_processed}, chunks={chunk_count}")
    console.print(f"[green]Chunking Completed[/green]: docs processed=[bold cyan]{docs_processed}[/bold cyan], chunks generated=[bold cyan]{chunk_count}[/bold cyan]")
    return report

# =========================
# CLI Entry
# =========================
def proceed() -> None:
    start = time.time()
    try:
        _ = run_chunking()
        elapsed = time.time() - start
        console.print(f"[green]Elapsed[/green]: {elapsed:.2f}s")
    except Exception as e:
        _log(f"FATAL: {e}")
        raise

