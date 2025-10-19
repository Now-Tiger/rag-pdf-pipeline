#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3
from typing import Optional, List, Dict, Any
from rich.console import Console
from pydantic import BaseModel

# from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, START

console = Console()

# ---------------------- LangGraph State Definition ----------------------
class ConversationState(BaseModel):
    user_input: str
    llm_response: str
    metadata: Dict[str, Any] = {}
    reranked_results: List[Dict[str, Any]] = []


# ---------------------- Persistent SQLite Store ----------------------
class SQLiteStore:
    def __init__(self, db_path: str = "chat_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            namespace TEXT,
            key TEXT UNIQUE,
            value TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        conn.close()

    def put(self, namespace: tuple[str] | str, key: str, state: ConversationState):
        """Persist state in SQLite."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        value_json = state.model_dump_json()
        ns = ".".join(namespace)
        c.execute("""
            INSERT OR REPLACE INTO conversations (namespace, key, value)
            VALUES (?, ?, ?)
        """, (ns, key, value_json))
        conn.commit()
        conn.close()

    def search(self, namespace: tuple[str] | str) -> List[ConversationState]:
        """Retrieve all states for a given namespace."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        ns = ".".join(namespace)
        c.execute("SELECT value FROM conversations WHERE namespace = ?", (ns,))
        rows = c.fetchall()
        conn.close()
        return [ConversationState.model_validate_json(r[0]) for r in rows]

# ---------------------- Memory Store Setup On Disk ----------------------
store = SQLiteStore()
# store: Any = InMemoryStore()

# ---------------------- Graph Compilation ----------------------
graph_builder: Any = StateGraph(state_schema=ConversationState)

# Add a simple placeholder node (e.g., conversation node) CV for conversation
graph_builder.add_node("CV", lambda state: state)

# Add an edge from START → conversation node (entrypoint)
graph_builder.add_edge(START, "CV")

graph = graph_builder.compile(store=store)

console.print("[green]✅ LangGraph memory initialized for RAG chat[/green]")


# ---------------------- Helper Functions ----------------------
def _safe_add_state(state: Any) -> None:
    """
    Store the given conversation state safely in the in-memory store.
    Each state is keyed uniquely to avoid overwriting.
    """
    try:
        namespace = "CV"
        # Get existing keys in this namespace
        existing = store.search(namespace) or []
        key = f"conv_{len(existing) + 1}"

        store.put(namespace, key, state)
        print(f"💾 Persisted memory state with key: {key}")

    except Exception as e:
        print(f"⚠️ Error persisting memory state: {e}")
        raise RuntimeError("⚠️ Unable to persist state.")


def _safe_list_states() -> List[Dict[str, Any]]:
    """
    Retrieve all stored states from the memory store.
    """
    for attr in ("get_states", "list_states", "all", "read_all"):
        target = getattr(store, attr, None)
        if callable(target):
            try:
                result = target()
                return _normalize_state_list(result)
            except Exception:
                continue
    return []


def _normalize_state_list(raw_states: Any) -> List[Dict[str, Any]]:
    """
    Normalize any raw object or dict list into List[Dict].
    """
    if raw_states is None:
        return []
    out: List[Dict[str, Any]] = []

    if isinstance(raw_states, list):
        for item in raw_states:
            if isinstance(item, dict):
                out.append(item)
            elif hasattr(item, "dict"):
                try:
                    out.append(item.dict())
                except Exception:
                    pass
            elif hasattr(item, "__dict__"):
                out.append(item.__dict__)
        return out

    if isinstance(raw_states, dict):
        return [raw_states]
    if hasattr(raw_states, "dict"):
        return [raw_states.dict()]
    return []


def add_interaction(
    user_input: str,
    llm_response: str,
    metadata: Optional[dict | list] = None,  # accept list too
    reranked_results: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Add a new conversation interaction to LangGraph memory.
    """
    # ----------------- Normalize metadata -----------------
    if isinstance(metadata, list):
        # Merge all dicts in list, fallback to empty dict if invalid
        combined_meta = {}
        for m in metadata:
            if isinstance(m, dict):
                combined_meta.update(m)
        metadata = combined_meta
    elif not isinstance(metadata, dict):
        metadata = {}

    state = ConversationState(
        user_input=user_input,
        llm_response=llm_response,
        metadata=metadata,
        reranked_results=reranked_results or []
    )

    _safe_add_state(state)
    console.print("[blue]📝 Interaction added to LangGraph memory[/blue]")


def get_conversation_history(limit: int = 10):
    namespace = ("conversations",)
    all_states = store.search(namespace)
    recent = all_states[-limit:]
    return [s.model_dump() for s in recent]


if __name__ == "__main__":
    # Demo run
    add_interaction("Who developed LangGraph?", "It’s by the LangChain team.")
    add_interaction("What is LangGraph used for?", "To build stateful LLM agents.")
    console.print(store.search("CV"))

    print("")
    history = get_conversation_history(5)
    console.print(history)



