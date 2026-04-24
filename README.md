# 🔎 Production-Grade RAG Pipeline

A modular, production-ready **Retrieval-Augmented Generation (RAG)** pipeline built with **LangChain LCEL**, **FAISS**, **HuggingFace embeddings**, **cross-encoder reranking**, and **Groq-hosted LLMs**.

This project goes well beyond a basic RAG demo. It includes hybrid retrieval (BM25 + dense vectors with Reciprocal Rank Fusion), semantic chunking, query rewriting, conversation memory, response caching, retry logic, async query execution, and built-in RAGAS-style evaluation of every answer.

---

## 📑 Table of Contents

1. [Features](#-features)
2. [Architecture](#-architecture)
3. [Pipeline Flow](#-pipeline-flow)
4. [Project Structure](#-project-structure)
5. [Installation](#-installation)
6. [Configuration](#-configuration)
7. [Usage](#-usage)
8. [Module Reference](#-module-reference)
9. [Configuration Flags](#-configuration-flags)
10. [Output & Logging](#-output--logging)
11. [Evaluation Metrics](#-evaluation-metrics)
12. [Troubleshooting](#-troubleshooting)
13. [Performance Notes](#-performance-notes)
14. [Roadmap](#-roadmap)

---

## ✨ Features

| Feature | Description |
|---|---|
| **Multi-format ingestion** | Load PDFs, plain text files, directories, and web URLs |
| **Semantic chunking** | Split documents on meaning boundaries, not arbitrary character counts |
| **Hybrid retrieval** | BM25 (keyword) + FAISS (dense vector) fused with Reciprocal Rank Fusion (RRF) |
| **Cross-encoder reranking** | Boost precision with a `ms-marco-MiniLM` reranker after retrieval |
| **Query rewriting** | LLM rewrites vague queries into retrieval-friendly form before searching |
| **Conversation memory** | Multi-turn chat with the last N turns injected into the prompt |
| **Response caching** | Disk-backed JSON cache — identical queries skip the LLM entirely |
| **Async execution** | Run multiple queries concurrently with `asyncio` |
| **Retry logic** | Exponential back-off on Groq rate limits and transient errors |
| **Structured logging** | Timestamped logs to both console and file (`rag_pipeline.log`) |
| **RAGAS-style evaluation** | Per-answer faithfulness + answer-relevancy scoring |
| **Source citations** | Every answer cites source filename and page number |
| **Source deduplication** | Clean, non-redundant source lists per response |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION LAYER                             │
│  PDFs   ·   TXT files   ·   Directories   ·   Web URLs              │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         CHUNKING LAYER                              │
│   Semantic Chunker (default)  →  Recursive Splitter (fallback)      │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         INDEXING LAYER                              │
│   HuggingFace Embeddings  →  FAISS Vector Store + BM25 Index        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL LAYER                              │
│   Query Rewrite → Hybrid Retriever (BM25 + FAISS via RRF)           │
│                  → Cross-Encoder Reranking                          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       GENERATION LAYER                              │
│   Cache Check → Prompt (context + history) → Groq LLM → Answer      │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       EVALUATION LAYER                              │
│   Faithfulness Scoring   ·   Answer Relevancy Scoring               │
│   Memory Update          ·   Cache Persistence                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Pipeline Flow

For each user question, the pipeline executes seven stages:

1. **Query Rewriting** — The LLM rewrites the question into a clearer, more retrieval-friendly form.
2. **Cache Lookup** — If this exact question + context combination has been seen, return the cached answer instantly.
3. **Hybrid Retrieval** — BM25 and FAISS each return their top-k candidates; results are fused using Reciprocal Rank Fusion.
4. **Reranking** — A cross-encoder rescores the merged candidates and keeps the most relevant top-N chunks.
5. **Generation** — The top chunks plus conversation history are injected into the prompt and sent to the Groq LLM.
6. **Caching & Evaluation** — The answer is saved to disk and scored on faithfulness and relevancy.
7. **Memory Update** — The Q&A pair is appended to conversation history for follow-up questions.

---

## 📂 Project Structure

```
RagStudyMaterial/
├── rag_pipeline.py          # Main pipeline (this file)
├── .env                     # GROQ_API_KEY (you create this)
├── rag_cache.json           # Auto-generated response cache
├── rag_pipeline.log         # Auto-generated structured log
├── faiss_index/             # Auto-generated FAISS vector index
│   ├── index.faiss
│   └── index.pkl
└── articles/                # Your source documents
    ├── World_war.pdf
    ├── Short_Novels_Introduction.pdf
    └── article.txt
```

---

## ⚙️ Installation

### Prerequisites

- **Python 3.10+**
- A free **Groq API key** from [console.groq.com](https://console.groq.com/keys)
- ~2 GB free disk space (for embedding models on first run)

### Install dependencies

```bash
pip install --upgrade \
    langchain \
    langchain-core \
    langchain-community \
    langchain-groq \
    langchain-experimental \
    langchain-huggingface \
    langchain-text-splitters \
    faiss-cpu \
    pymupdf \
    sentence-transformers \
    python-dotenv \
    rank-bm25 \
    tenacity
```

> 💡 On the first run, HuggingFace will download the embedding model (`all-MiniLM-L6-v2`, ~80 MB) and the cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`, ~80 MB). This happens once and is cached locally.

---

## 🔐 Configuration

### Step 1 — Create a `.env` file

In the project root, create a file named `.env`:

```env
GROQ_API_KEY=gsk_your_actual_key_here
```

### Step 2 — Load the key from environment (recommended)

In `create_llm()`, replace the hardcoded key with:

```python
api_key = os.getenv("GROQ_API_KEY")
```

> ⚠️ **Security warning:** The current source contains a hardcoded API key. **Rotate that key immediately** at [console.groq.com/keys](https://console.groq.com/keys) and switch to the `.env` approach above. Hardcoded secrets are a serious security risk — anyone with access to your code can drain your quota.

### Step 3 — Update document paths

In `__main__`, set `SAMPLE_DOCUMENTS` to your file paths:

```python
SAMPLE_DOCUMENTS = [
    "path/to/your/document1.pdf",
    "path/to/your/document2.txt",
]
```

---

## 🚀 Usage

### Basic run

```bash
python rag_pipeline.py
```

### Programmatic usage

```python
from rag_pipeline import run_full_rag_pipeline

run_full_rag_pipeline(
    document_paths = ["docs/research_paper.pdf"],
    web_urls       = ["https://example.com/article"],
    questions      = [
        "What is the main contribution of this paper?",
        "What evaluation metrics did the authors use?",
    ],
    rebuild_store         = True,
    use_semantic_chunking = True,
    use_hybrid_retriever  = True,
    use_reranker          = True,
    evaluate_answers      = True,
    rewrite_queries       = True,
    use_cache             = True,
    use_memory            = True,
    run_async             = False,
    debug                 = False,
)
```

### Reusing an existing index

After your first run, set `rebuild_store=False` to skip re-indexing:

```python
run_full_rag_pipeline(
    questions     = ["A new question about the same docs"],
    rebuild_store = False,    # loads cached FAISS index
)
```

### Concurrent queries

For batch workloads, enable async to run all questions in parallel:

```python
run_full_rag_pipeline(
    document_paths = SAMPLE_DOCUMENTS,
    questions      = SAMPLE_QUESTIONS,
    run_async      = True,
)
```

---

## 📚 Module Reference

### Document Loaders

| Function | Purpose |
|---|---|
| `load_documents_from_pdf(path)` | Load a single PDF via PyMuPDF |
| `load_documents_from_text(path)` | Load a `.txt` file |
| `load_documents_from_directory(dir, glob)` | Recursively load files matching a glob |
| `load_documents_from_web(urls)` | Scrape web pages |

### Chunking

| Function | Purpose |
|---|---|
| `split_documents_semantic(docs, embeddings)` | Meaning-aware semantic chunking |
| `split_documents_recursive(docs)` | Fallback fixed-size character splitter |

### Indexing & Retrieval

| Function / Class | Purpose |
|---|---|
| `create_embeddings_model(name)` | Load a HuggingFace sentence-transformer |
| `build_faiss_vectorstore(chunks, embeddings)` | Build and persist a FAISS index |
| `load_faiss_vectorstore(embeddings, path)` | Load a saved index from disk |
| `HybridRetriever` | BM25 + FAISS fused with RRF |
| `create_hybrid_retriever(chunks, vs, k)` | Factory for `HybridRetriever` |

### Reranking & Generation

| Function | Purpose |
|---|---|
| `load_reranker(model)` | Load a cross-encoder reranker |
| `rerank_documents(reranker, query, docs, top_n)` | Score and reorder retrieved chunks |
| `create_llm(model, temp, max_tokens)` | Instantiate the Groq LLM |
| `invoke_with_retry(llm, prompt)` | LLM call with exponential back-off |
| `rewrite_query(llm, question)` | LLM-based query rewriting |

### Orchestration & Utilities

| Function / Class | Purpose |
|---|---|
| `ResponseCache` | Disk-backed answer cache |
| `ConversationMemory` | Sliding-window chat history |
| `create_rag_prompt()` | Build the main RAG prompt template |
| `build_rag_chain(llm, retriever, prompt)` | Assemble the LCEL chain |
| `evaluate_response(llm, q, a, ctx)` | Faithfulness + relevancy scoring |
| `query_rag_pipeline(...)` | Full single-query pipeline |
| `query_rag_pipeline_async(...)` | Async variant |
| `run_queries_concurrently(...)` | Run a list of queries in parallel |
| `run_full_rag_pipeline(...)` | End-to-end orchestrator |

---

## 🎛 Configuration Flags

Flags accepted by `run_full_rag_pipeline`:

| Flag | Default | Description |
|---|---|---|
| `document_paths` | `None` | List of local `.pdf` / `.txt` paths |
| `web_urls` | `None` | List of URLs to scrape |
| `questions` | `None` | List of questions to answer |
| `rebuild_store` | `True` | Rebuild FAISS index from scratch |
| `use_semantic_chunking` | `True` | Use meaning-aware chunking |
| `use_hybrid_retriever` | `True` | BM25 + FAISS with RRF fusion |
| `use_reranker` | `True` | Apply cross-encoder reranking |
| `evaluate_answers` | `True` | Score faithfulness + relevancy |
| `rewrite_queries` | `True` | LLM rewrites queries before retrieval |
| `use_cache` | `True` | Cache answers to `rag_cache.json` |
| `use_memory` | `True` | Remember previous Q&A turns |
| `run_async` | `False` | Run all questions concurrently |
| `debug` | `False` | Print raw retrieved chunks |

---

## 📊 Output & Logging

For each query, the console prints:

```
============================================================
[Query] What is artificial intelligence?
============================================================

[Answer] (3.2s)
Artificial intelligence (AI) refers to computer systems that can perform tasks
typically requiring human intelligence, such as reasoning, learning, perception,
and decision-making [Source: article.txt].

[Sources]
  • article.txt
  • World_war.pdf, p.7

[Eval] Faithfulness=0.95 | Relevancy=0.92
```

After all queries finish, an evaluation summary is printed:

```
============================================================
          EVALUATION SUMMARY
============================================================
  Questions answered   : 5
  Cache hits           : 0
  Avg faithfulness     : 0.93 / 1.00
  Avg answer relevancy : 0.89 / 1.00
  Avg latency          : 3.4s per query
============================================================
```

All events are also written to `rag_pipeline.log` with timestamps and module names for debugging and audit trails.

---

## 📈 Evaluation Metrics

Inspired by [RAGAS](https://github.com/explodinggradients/ragas), the pipeline scores every answer on two dimensions using the LLM as a judge:

| Metric | What It Measures | Range |
|---|---|---|
| **Faithfulness** | Are all claims in the answer supported by the retrieved context? | 0.0 – 1.0 |
| **Answer Relevancy** | Does the answer actually address the question asked? | 0.0 – 1.0 |

A score of `0.0` means complete failure (hallucinated / off-topic), `1.0` means a perfectly grounded and on-topic response.

These metrics help you quickly spot retrieval failures, prompt issues, or chunks that need to be re-split.

---

## 🛠 Troubleshooting

### `GROQ_API_KEY not found`

You haven't created the `.env` file or it's not in the working directory. Verify with:

```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('GROQ_API_KEY'))"
```

### `FAISS index not found`

You set `rebuild_store=False` but no index exists yet. Run once with `rebuild_store=True` to create it.

### Few semantic chunks produced

The pipeline auto-falls back to the recursive splitter when fewer than 5 semantic chunks are produced. This usually means your documents are very short. You can force the recursive splitter with `use_semantic_chunking=False`.

### Slow first run

The first run downloads embedding and reranker models (~160 MB total). Subsequent runs use the local cache and are much faster.

### Rate limit errors from Groq

The `invoke_with_retry` decorator handles transient failures with exponential back-off (2s → 4s → 8s, 3 attempts). For sustained workloads, lower your concurrency or upgrade your Groq plan.

### Cached answers seem stale

Either delete `rag_cache.json` manually or call `ResponseCache().clear()` at startup (already done in `__main__`).

---

## ⚡ Performance Notes

- **Embedding model choice:** `all-MiniLM-L6-v2` (default, 384-dim, fast) vs `all-mpnet-base-v2` (768-dim, more accurate, ~3x slower).
- **`k` for retrieval:** 8 by default. Higher recall, lower precision. Reranking compensates by keeping only the best 4.
- **Reranker on/off:** Reranking adds ~100–300 ms per query but typically lifts faithfulness scores noticeably.
- **Async mode:** Best when answering many independent questions. Not useful for multi-turn conversations (memory becomes ambiguous).
- **Caching:** A cache hit returns in milliseconds. Clearing the cache forces re-evaluation.

---

## 🗺 Roadmap

Potential future enhancements:

- Streaming responses (token-by-token output)
- Re-ranking with a second LLM pass instead of cross-encoder
- HyDE (Hypothetical Document Embeddings) for vague queries
- Multi-vector retrieval (parent-document strategy)
- Pluggable vector stores (Qdrant, Weaviate, Chroma, Pinecone)
- Web UI (Streamlit / Gradio)
- Containerization (Dockerfile + compose)
- Unit tests with `pytest`
- CI pipeline with GitHub Actions

---

## 📝 License

Add your preferred license here (MIT recommended for open-source projects).

---

## 🙏 Acknowledgements

Built on top of these excellent open-source projects:

- [LangChain](https://github.com/langchain-ai/langchain) — RAG orchestration framework
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [Sentence-Transformers](https://www.sbert.net/) — Embeddings & cross-encoders
- [Groq](https://groq.com/) — Fast LLM inference
- [RAGAS](https://github.com/explodinggradients/ragas) — Evaluation methodology inspiration
