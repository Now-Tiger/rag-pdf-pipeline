# **Technical Documentation: RAG PDF Pipeline with Multilingual Support & Chat Memory**

---

## **1. System Overview**

The RAG PDF Pipeline is a **Retrieval-Augmented Generation system** designed to answer natural language questions over multilingual PDF documents. The system combines **vector-based retrieval**, **metadata filtering**, **LLM-powered generation**, and **persistent chat memory**.

**Key Objectives:**

- Enable semantic search across multiple languages (English, Chinese, Urdu, Bengali).
- Maintain multi-turn chat memory to support contextual question answering.
- Provide persistent storage to survive system restarts.

**High-Level Architecture:**

```
User Query → Query Embedding → Vector DB Retrieval → Metadata Filter → Reranking → LLM Answer Generation → Chat Memory Storage → Response
```

---

## **2. Components**

| Component                      | Description                                                                                                                          |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Query CLI**                  | Interactive command-line interface for user queries with optional metadata filters (language, source PDF).                           |
| **Embedding Module**           | Uses `SentenceTransformer` (`intfloat/multilingual-e5-large`) to create normalized vector embeddings for semantic search.            |
| **Vector Database (ChromaDB)** | Stores PDF chunks with embeddings, metadata, and supports efficient similarity queries.                                              |
| **Query Decomposer**           | Breaks complex queries into sub-queries for better retrieval accuracy.                                                               |
| **Metadata Filter**            | Filters retrieved chunks based on language and source PDF.                                                                           |
| **Reranker**                   | Orders top retrieved chunks based on similarity and relevance scores.                                                                |
| **LLM Module**                 | Gemini-2.5-Flash model generates answers using top-k reranked chunks, supports multilingual responses and English translation.       |
| **Chat Memory**                | Persistent conversation memory using **SQLite-backed LangGraph**. Stores user inputs, LLM responses, metadata, and reranked results. |
| **PDF Parser & OCR**           | Extracts text from PDFs. Uses **Tesseract OCR** for non-digital PDFs. Issues exist for Urdu/Bengali OCR with free tools.             |

---

## **3. Data Flow**

1. **User Input**: User enters a query via CLI, optionally specifying language or PDF filters.
2. **Embedding Generation**: Query is converted to embedding vector.
3. **Vector Retrieval**: Top-K similar chunks retrieved from ChromaDB.
4. **Metadata Filtering**: Optional filters applied for language and source PDF.
5. **Reranking**: Retrieved chunks are scored and sorted.
6. **LLM Answer Generation**: Context from top chunks is fed to Gemini model to generate multilingual answer.
7. **Chat Memory Persistence**: Interaction stored in SQLite via LangGraph, summarizing older states and keeping recent states intact.
8. **Response**: Final answer returned to CLI with optional English translation.

---

## **4. Persistent Chat Memory Architecture**

```
ConversationState
├─ user_input: str
├─ llm_response: str
├─ metadata: Dict[str, Any]
└─ reranked_results: List[Dict[str, Any]]
```

- **SQLiteStore**: Provides persistent storage on disk.
- **LangGraph StateGraph**: Maintains conversation state transitions.
- **Graph Nodes & Edges**:
  - Entry node: `START`
  - Conversation node: `CV`
  - Edge: `START → CV`

- Supports retrieval of last N interactions for context-aware LLM responses.

---

## **5. System Diagram**

```
   ┌─────────────┐
   │   User CLI  │
   └─────┬───────┘
         │ Query Input
         ▼
   ┌─────────────┐
   │  Query Embed│
   └─────┬───────┘
         │ Embedding Vector
         ▼
   ┌─────────────┐
   │ Vector DB   │ ←─ PDF Chunks with Metadata
   └─────┬───────┘
         │ Top-K Chunks
         ▼
   ┌─────────────┐
   │ Metadata    │
   │ Filtering   │
   └─────┬───────┘
         │ Filtered Chunks
         ▼
   ┌─────────────┐
   │   Reranker  │
   └─────┬───────┘
         │ Top-k Chunks
         ▼
   ┌─────────────┐
   │    LLM      │
   │ Gemini-2.5  │
   └─────┬───────┘
         │ Generated Answer
         ▼
   ┌─────────────┐
   │ Chat Memory │
   │  SQLite     │
   └─────┬───────┘
         │ Persisted Conversation
         ▼
   ┌─────────────┐
   │   Response  │
   └─────────────┘
```

---

## **6. Key Packages & Dependencies**

- Python packages:
  - `langgraph` (state & memory management)
  - `sentence-transformers` (multilingual embeddings)
  - `chromadb` (vector DB)
  - `rich` (CLI enhancements)
  - `pydantic` (data validation)

- System dependencies:
  - **Tesseract OCR** (required for PyTesseract PDF parsing)

- Optional: `pdfplumber`, `PyPDF2` for PDF text extraction.

---

## **7. Limitations & Future Enhancements**

- **Limitations**:
  - Free OCR (Tesseract) struggles with Urdu/Bengali PDFs.
  - Non-digital PDFs with poor quality affect embeddings and retrieval.

- **Future Enhancements**:
  1. **Better OCR**: Integrate commercial OCR (Surya OCR) for improved Urdu/Bengali parsing.
  2. **Vector DB Scaling**: Move to Weaviate/Milvus for large datasets.
  3. **Advanced Chunking**: Semantic-aware chunk merging.
  4. **Domain-tuned LLMs**: Fine-tune LLMs for higher relevance.
  5. **Multi-turn QA**: Contextual follow-up queries with chat memory.
