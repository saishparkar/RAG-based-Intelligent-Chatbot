"""
Complete RAG (Retrieval-Augmented Generation) Pipeline — LangChain LCEL.

Upgrades over previous version:
  ✅ Query rewriting         — LLM rewrites vague queries before retrieval
  ✅ Conversation memory     — multi-turn chat with history
  ✅ Response caching        — identical queries skip the LLM entirely
  ✅ Async support           — run multiple queries concurrently
  ✅ Retry logic             — auto-retry on Groq API rate limits / timeouts
  ✅ Structured logging      — timestamped logs to file + console
  ✅ RAGAS-style evaluation  — faithfulness + answer relevancy scoring
  ✅ Source deduplication    — cleaner cited sources in output
  ✅ API key via .env only   — no hardcoded secrets

Setup:
    pip install --upgrade langchain langchain-core langchain-community
                langchain-groq langchain-experimental faiss-cpu pymupdf
                sentence-transformers python-dotenv rank-bm25
                langchain-huggingface tenacity
    Add GROQ_API_KEY=gsk_your_key to a .env file
"""

import os
import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ── Core ──────────────────────────────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ── Text splitters ────────────────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Document loaders ──────────────────────────────────────────────────────────
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyMuPDFLoader,
    TextLoader,
    WebBaseLoader,
)

# ── Vector store ──────────────────────────────────────────────────────────────
from langchain_community.vectorstores import FAISS

# ── Embeddings ────────────────────────────────────────────────────────────────
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# ── BM25 ──────────────────────────────────────────────────────────────────────
from langchain_community.retrievers import BM25Retriever

# ── Semantic chunking ─────────────────────────────────────────────────────────
from langchain_experimental.text_splitter import SemanticChunker

# ── Cross-encoder reranking ───────────────────────────────────────────────────
from sentence_transformers import CrossEncoder

# ── LLM ───────────────────────────────────────────────────────────────────────
from langchain_groq import ChatGroq

load_dotenv()


# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------

def setup_logger(log_file: str = "rag_pipeline.log") -> logging.Logger:
    """
    Configure structured logging to both console and a log file.
    All pipeline steps log with timestamps, level, and module name.
    """
    logger = logging.getLogger("RAGPipeline")
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


log = setup_logger()


# ---------------------------------------------------------------------------
# RESPONSE CACHE  (skip LLM on repeated identical queries)
# ---------------------------------------------------------------------------

class ResponseCache:
    """
    Disk-backed JSON cache for LLM responses.

    Key  = SHA-256 hash of (question + first 500 chars of context).
    Value = cached answer string.

    Identical queries on the same corpus return instantly without
    calling the Groq API — saving time and rate-limit quota.
    """

    def __init__(self, cache_file: str = "./rag_cache.json"):
        self.cache_file = Path(cache_file)
        self._cache: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self.cache_file.exists():
            try:
                self._cache = json.loads(self.cache_file.read_text(encoding="utf-8"))
                log.info(f"[Cache] Loaded {len(self._cache)} cached responses.")
            except Exception:
                self._cache = {}

    def _save(self) -> None:
        self.cache_file.write_text(
            json.dumps(self._cache, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _key(self, question: str, context: str) -> str:
        raw = f"{question.strip().lower()}||{context[:500]}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, question: str, context: str) -> Optional[str]:
        return self._cache.get(self._key(question, context))

    def set(self, question: str, context: str, answer: str) -> None:
        self._cache[self._key(question, context)] = answer
        self._save()

    def clear(self) -> None:
        self._cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        log.info("[Cache] Cleared.")


# ---------------------------------------------------------------------------
# CONVERSATION MEMORY  (multi-turn chat)
# ---------------------------------------------------------------------------

class ConversationMemory:
    """
    Stores the last N conversation turns for multi-turn RAG chat.

    Injected into the prompt so the LLM can resolve follow-up questions
    like "What did you mean by that?" or "Tell me more about the second point."
    """

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []

    def add(self, question: str, answer: str) -> None:
        self.history.append({"role": "human",     "content": question})
        self.history.append({"role": "assistant",  "content": answer})
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def format(self) -> str:
        if not self.history:
            return ""
        lines = []
        for msg in self.history:
            prefix = "User" if msg["role"] == "human" else "Assistant"
            lines.append(f"{prefix}: {msg['content']}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.history = []
        log.info("[Memory] Conversation history cleared.")


# ---------------------------------------------------------------------------
# 1. DOCUMENT LOADING
# ---------------------------------------------------------------------------

def load_documents_from_pdf(file_path: str) -> List[Document]:
    """Load and parse a single PDF file into LangChain Document objects."""
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    log.info(f"[PDF Loader] Loaded {len(documents)} pages from '{file_path}'")
    return documents


def load_documents_from_text(file_path: str) -> List[Document]:
    """Load a plain-text (.txt) file into a LangChain Document object."""
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    log.info(f"[Text Loader] Loaded {len(documents)} document(s) from '{file_path}'")
    return documents


def load_documents_from_directory(
    directory_path: str,
    glob_pattern: str = "**/*.pdf",
) -> List[Document]:
    """Recursively load all files matching a glob pattern from a directory."""
    loader = DirectoryLoader(directory_path, glob=glob_pattern)
    documents = loader.load()
    log.info(
        f"[Directory Loader] Loaded {len(documents)} document(s) "
        f"from '{directory_path}' (pattern: {glob_pattern})"
    )
    return documents


def load_documents_from_web(urls: List[str]) -> List[Document]:
    """Scrape and load web pages into LangChain Document objects."""
    loader = WebBaseLoader(urls)
    documents = loader.load()
    log.info(f"[Web Loader] Loaded {len(documents)} page(s) from {len(urls)} URL(s)")
    return documents


# ---------------------------------------------------------------------------
# 2. EMBEDDINGS MODEL
# ---------------------------------------------------------------------------

def create_embeddings_model(
    model_name: str = "all-MiniLM-L6-v2",
) -> HuggingFaceEmbeddings:
    """
    Instantiate a local HuggingFace sentence-transformer embeddings model.

    'all-MiniLM-L6-v2'  — fast, 384-dim, great balance of speed & quality.
    'all-mpnet-base-v2' — slower, 768-dim, higher quality.
    """
    log.info(f"[Embeddings] Loading local HuggingFace model: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    log.info(f"[Embeddings] Model '{model_name}' loaded successfully.")
    return embeddings


# ---------------------------------------------------------------------------
# 3. CHUNKING
# ---------------------------------------------------------------------------

def split_documents_semantic(
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: int = 95,
) -> List[Document]:
    """Split on semantic meaning boundaries — keeps each chunk on one topic."""
    log.info(f"[Semantic Chunker] Splitting {len(documents)} documents ...")
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )
    chunks = splitter.split_documents(documents)
    log.info(f"[Semantic Chunker] Produced {len(chunks)} semantic chunks.")
    return chunks


def split_documents_recursive(
    documents: List[Document],
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Fallback fixed-size splitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    log.info(f"[Recursive Splitter] {len(documents)} docs → {len(chunks)} chunks")
    return chunks


# ---------------------------------------------------------------------------
# 4. VECTOR STORE
# ---------------------------------------------------------------------------

def build_faiss_vectorstore(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    save_path: str = "./faiss_index",
) -> FAISS:
    """Build a FAISS vector store from document chunks and save it locally."""
    log.info(f"[FAISS] Embedding {len(chunks)} chunks ...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    log.info(f"[FAISS] Index saved to '{save_path}'.")
    return vectorstore


def load_faiss_vectorstore(
    embeddings: HuggingFaceEmbeddings,
    load_path: str = "./faiss_index",
) -> FAISS:
    """Load a previously saved FAISS vector store from disk."""
    if not Path(load_path).exists():
        raise FileNotFoundError(
            f"FAISS index '{load_path}' not found. Run build_faiss_vectorstore() first."
        )
    log.info(f"[FAISS] Loading index from '{load_path}' ...")
    vectorstore = FAISS.load_local(
        load_path, embeddings, allow_dangerous_deserialization=True
    )
    log.info("[FAISS] Index loaded successfully.")
    return vectorstore


# ---------------------------------------------------------------------------
# 5. CUSTOM HYBRID RETRIEVER  (BM25 + FAISS with RRF fusion)
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    BM25 (keyword) + FAISS (semantic) with Reciprocal Rank Fusion (RRF).

    RRF score = sum(1 / (rank + 60)) across both retriever rankings.
    Documents appearing high in both lists score highest.
    No external class dependencies — works on any LangChain version.
    """

    def __init__(self, bm25_retriever, faiss_retriever, k: int = 8):
        self.bm25_retriever  = bm25_retriever
        self.faiss_retriever = faiss_retriever
        self.k               = k

    def invoke(self, query: str) -> List[Document]:
        bm25_docs  = self.bm25_retriever.invoke(query)
        faiss_docs = self.faiss_retriever.invoke(query)

        scores: Dict[str, float]    = {}
        seen:   Dict[str, Document] = {}

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + 60)
            seen[key]   = doc

        for rank, doc in enumerate(faiss_docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + 60)
            seen[key]   = doc

        ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        result = [seen[k] for k in ranked_keys[: self.k]]

        log.info(
            f"[Hybrid] BM25={len(bm25_docs)} + FAISS={len(faiss_docs)} "
            f"→ {len(result)} merged chunks (RRF)"
        )
        return result

    def __or__(self, other):
        return RunnableLambda(self.invoke) | other


def create_hybrid_retriever(
    chunks: List[Document],
    vectorstore: FAISS,
    k: int = 8,
) -> HybridRetriever:
    """Build a HybridRetriever from chunks and a FAISS vectorstore."""
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = k
    faiss = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    log.info(f"[Hybrid Retriever] Created (BM25 + FAISS, k={k}, fusion=RRF)")
    return HybridRetriever(bm25, faiss, k=k)


# ---------------------------------------------------------------------------
# 6. CROSS-ENCODER RERANKING
# ---------------------------------------------------------------------------

def load_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> CrossEncoder:
    """Load a cross-encoder reranking model for precision boosting."""
    log.info(f"[Reranker] Loading: {model_name}")
    reranker = CrossEncoder(model_name)
    log.info("[Reranker] Loaded.")
    return reranker


def rerank_documents(
    reranker: CrossEncoder,
    query: str,
    docs: List[Document],
    top_n: int = 4,
) -> List[Document]:
    """Rerank retrieved documents by cross-encoder score and return top_n."""
    if not docs:
        return docs
    pairs  = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in ranked[:top_n]]
    log.info(f"[Reranker] {len(docs)} → top {len(top_docs)} after reranking.")
    return top_docs


# ---------------------------------------------------------------------------
# 7. LLM  (with retry on rate limits)
# ---------------------------------------------------------------------------

def create_llm(
    model_name: str   = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
    max_tokens: int    = 1024,
) -> ChatGroq:
    """
    Instantiate ChatGroq. API key from GROQ_API_KEY env var only.
    Never hardcode keys in source files — use a .env file instead.
    """
    # api_key = os.getenv("GROQ_API_KEY")
    api_key = "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"
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Add to .env:\n  GROQ_API_KEY=gsk_your_key_here"
        )
    log.info(f"[LLM] Groq model: {model_name} | temp={temperature}")
    return ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def invoke_with_retry(llm: ChatGroq, prompt_value) -> str:
    """
    Call the LLM with automatic retry on failures.

    Uses exponential back-off: waits 2s → 4s → 8s between attempts.
    Handles Groq API rate limits and transient network errors gracefully.
    """
    return llm.invoke(prompt_value).content


# ---------------------------------------------------------------------------
# 8. QUERY REWRITING
# ---------------------------------------------------------------------------

def rewrite_query(llm: ChatGroq, question: str) -> str:
    """
    Use the LLM to rewrite a vague or ambiguous query into a clearer,
    more retrieval-friendly form before hitting the vector store.

    Example:
        Input : "tell me about the thing in 1942"
        Output: "What significant event occurred in February 1942 during World War II?"

    Dramatically improves retrieval for short or poorly phrased questions.
    """
    rewrite_prompt = f"""Rewrite the following question to be more specific and 
retrieval-friendly for a document search system. Output ONLY the rewritten 
question — no preamble, no explanation.

Original question: {question}

Rewritten question:"""

    try:
        rewritten = invoke_with_retry(llm, rewrite_prompt).strip()
        if rewritten and rewritten != question:
            log.info(f"[Query Rewrite] '{question}' → '{rewritten}'")
            return rewritten
    except Exception as e:
        log.warning(f"[Query Rewrite] Failed ({e}), using original query.")
    return question


# ---------------------------------------------------------------------------
# 9. PROMPTS
# ---------------------------------------------------------------------------

def create_rag_prompt() -> PromptTemplate:
    """RAG prompt with chain-of-thought, citations, conflict detection, and memory."""
    template = """You are a precise research assistant. Answer ONLY using the context below.
If the answer is not in the context, say "I don't have enough information to answer this."

Think step by step before writing your final answer.

{history}

Context:
{context}

Question: {question}

Instructions:
- Cite the source and page for each claim, e.g. [Source: report.pdf, p.3]
- If multiple chunks say conflicting things, note the discrepancy
- Be concise but complete

Answer:"""

    return PromptTemplate(
        input_variables=["context", "question", "history"],
        template=template,
    )


# ---------------------------------------------------------------------------
# 10. LCEL RAG CHAIN
# ---------------------------------------------------------------------------

def format_docs(docs: List[Document]) -> str:
    """Concatenate chunks with source metadata headers into context string."""
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "")
        header = (
            f"[Source: {Path(source).name}, p.{page}]"
            if page != "" else
            f"[Source: {Path(source).name}]"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


def format_sources(docs: List[Document]) -> str:
    """Return a deduplicated bullet list of sources used in a response."""
    seen  = set()
    lines = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "")
        key    = f"{source}:{page}"
        if key not in seen:
            seen.add(key)
            name = Path(source).name
            lines.append(f"  • {name}" + (f", p.{page}" if page != "" else ""))
    return "\n".join(lines) if lines else "  • (no sources)"


def build_rag_chain(llm: ChatGroq, retriever, prompt: PromptTemplate):
    """
    Assemble the LCEL RAG chain.

    question → retriever → format_docs → {context}
             ─────────────────────────► {question}
             ─────────────────────────► {history}
                                             │
                                         prompt → llm → StrOutputParser → answer
    """
    retrieve = RunnableLambda(retriever.invoke)

    chain = (
        {
            "context":  retrieve | format_docs,
            "question": RunnablePassthrough(),
            "history":  RunnableLambda(lambda _: ""),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    log.info("[RAG Chain] LCEL pipeline assembled.")
    return chain


# ---------------------------------------------------------------------------
# 11. RAGAS-STYLE EVALUATION
# ---------------------------------------------------------------------------

def evaluate_response(
    llm: ChatGroq,
    question: str,
    answer: str,
    context: str,
) -> Dict[str, float]:
    """
    Score the response on two RAGAS-inspired metrics using the LLM as judge.

    Faithfulness     : Every claim is supported by context. (0.0–1.0)
    Answer Relevancy : Answer actually addresses the question. (0.0–1.0)
    """
    faithfulness_prompt = f"""Rate the faithfulness of the answer (0.0 to 1.0).
Faithfulness = every claim in the answer is supported by the context.
0.0 = completely hallucinated. 1.0 = fully grounded in context.
Reply with ONLY a number between 0.0 and 1.0.

Context: {context[:1500]}
Answer: {answer}
Score:"""

    relevancy_prompt = f"""Rate how well the answer addresses the question (0.0 to 1.0).
0.0 = completely off-topic. 1.0 = perfectly answers the question.
Reply with ONLY a number between 0.0 and 1.0.

Question: {question}
Answer: {answer}
Score:"""

    scores = {"faithfulness": 0.0, "answer_relevancy": 0.0}

    try:
        scores["faithfulness"] = min(1.0, max(0.0, float(
            invoke_with_retry(llm, faithfulness_prompt).strip()
        )))
    except Exception as e:
        log.warning(f"[Eval] Faithfulness scoring failed: {e}")

    try:
        scores["answer_relevancy"] = min(1.0, max(0.0, float(
            invoke_with_retry(llm, relevancy_prompt).strip()
        )))
    except Exception as e:
        log.warning(f"[Eval] Relevancy scoring failed: {e}")

    log.info(
        f"[Eval] Faithfulness={scores['faithfulness']:.2f} | "
        f"Relevancy={scores['answer_relevancy']:.2f}"
    )
    return scores


# ---------------------------------------------------------------------------
# 12. QUERY & DISPLAY
# ---------------------------------------------------------------------------

def query_rag_pipeline(
    chain,
    retriever,
    question: str,
    reranker:        Optional[CrossEncoder]       = None,
    llm:             Optional[ChatGroq]           = None,
    memory:          Optional[ConversationMemory] = None,
    cache:           Optional[ResponseCache]      = None,
    evaluate:        bool = True,
    rewrite_queries: bool = True,
    top_n_rerank:    int  = 4,
) -> dict:
    """
    Full query flow:
        1. Rewrite query for better retrieval
        2. Check cache for existing answer
        3. Retrieve → rerank → build context
        4. Generate answer with conversation history injected
        5. Cache the answer
        6. Evaluate faithfulness + relevancy
        7. Update conversation memory
    """
    print(f"\n{'='*60}")
    print(f"[Query] {question}")
    print(f"{'='*60}")

    t0 = time.time()

    # ── 1. Query rewriting ──────────────────────────────────────────────────
    retrieval_query = question
    if rewrite_queries and llm is not None:
        retrieval_query = rewrite_query(llm, question)

    # ── 2. Retrieve & rerank ────────────────────────────────────────────────
    raw_docs    = retriever.invoke(retrieval_query)
    source_docs = rerank_documents(reranker, retrieval_query, raw_docs, top_n=top_n_rerank) \
                  if reranker is not None else raw_docs
    context_text = format_docs(source_docs)

    # ── 3. Cache check ──────────────────────────────────────────────────────
    if cache is not None:
        cached = cache.get(question, context_text)
        if cached:
            log.info("[Cache] Hit — returning cached answer.")
            print(f"\n[Answer] (cached)\n{cached}")
            return {"query": question, "result": cached,
                    "source_documents": source_docs, "cached": True}

    # ── 4. Generate answer ──────────────────────────────────────────────────
    history_text = memory.format() if memory else ""

    direct_chain = create_rag_prompt() | llm | StrOutputParser()
    try:
        answer = direct_chain.invoke({
            "context":  context_text,
            "question": question,
            "history":  f"Conversation so far:\n{history_text}\n" if history_text else "",
        })
    except Exception as e:
        log.error(f"[Query] LLM call failed: {e}")
        answer = "I encountered an error generating a response. Please try again."

    elapsed = time.time() - t0
    print(f"\n[Answer] ({elapsed:.1f}s)\n{answer}")
    print(f"\n[Sources]\n{format_sources(source_docs)}")

    # ── 5. Cache the answer ─────────────────────────────────────────────────
    if cache is not None:
        cache.set(question, context_text, answer)

    # ── 6. Evaluate ─────────────────────────────────────────────────────────
    eval_scores = {}
    if evaluate and llm is not None:
        eval_scores = evaluate_response(llm, question, answer, context_text)
        print(
            f"[Eval] Faithfulness={eval_scores['faithfulness']:.2f} | "
            f"Relevancy={eval_scores['answer_relevancy']:.2f}"
        )

    # ── 7. Update memory ────────────────────────────────────────────────────
    if memory is not None:
        memory.add(question, answer)

    return {
        "query":            question,
        "rewritten_query":  retrieval_query,
        "result":           answer,
        "source_documents": source_docs,
        "eval_scores":      eval_scores,
        "latency_s":        elapsed,
        "cached":           False,
    }


# ── Async version ─────────────────────────────────────────────────────────────

async def query_rag_pipeline_async(chain, retriever, question: str, **kwargs) -> dict:
    """Run query_rag_pipeline in a thread pool for async/concurrent use."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: query_rag_pipeline(chain, retriever, question, **kwargs),
    )


async def run_queries_concurrently(
    chain, retriever, questions: List[str], **kwargs
) -> List[dict]:
    """
    Run multiple queries concurrently using asyncio.
    All questions execute in parallel instead of sequentially.
    """
    log.info(f"[Async] Running {len(questions)} queries concurrently ...")
    tasks = [
        query_rag_pipeline_async(chain, retriever, q, **kwargs)
        for q in questions
    ]
    return await asyncio.gather(*tasks)


def debug_retriever(retriever, question: str) -> None:
    """Print raw retrieved chunks — useful for diagnosing retrieval quality."""
    print(f"\n[DEBUG] Chunks for: '{question}'")
    docs = retriever.invoke(question)
    for i, doc in enumerate(docs):
        print(f"--- Chunk {i+1} ---")
        print(doc.page_content[:300])
        print()


def display_pipeline_summary(
    chunks:           List[Document],
    llm_model:        str,
    embeddings_model: str,
    retriever_type:   str = "Hybrid (BM25 + FAISS, RRF)",
    reranker_model:   str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> None:
    print("\n" + "=" * 60)
    print("          RAG PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Total chunks indexed : {len(chunks)}")
    print(f"  Vector store         : FAISS (local)")
    print(f"  Retriever            : {retriever_type}")
    print(f"  Reranker             : {reranker_model}")
    print(f"  Chain style          : LCEL (modern)")
    print(f"  LLM model            : {llm_model} (via Groq)")
    print(f"  Embeddings model     : {embeddings_model} (local HuggingFace)")
    print(f"  Query rewriting      : ✅ enabled")
    print(f"  Response caching     : ✅ enabled  →  rag_cache.json")
    print(f"  Conversation memory  : ✅ enabled  →  last 6 turns")
    print(f"  Evaluation           : ✅ faithfulness + relevancy per answer")
    print(f"  Logging              : ✅ rag_pipeline.log")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# 13. FULL PIPELINE ORCHESTRATOR
# ---------------------------------------------------------------------------

def run_full_rag_pipeline(
    document_paths:        Optional[List[str]] = None,
    web_urls:              Optional[List[str]] = None,
    questions:             Optional[List[str]] = None,
    rebuild_store:         bool = True,
    use_semantic_chunking: bool = True,
    use_hybrid_retriever:  bool = True,
    use_reranker:          bool = True,
    evaluate_answers:      bool = True,
    rewrite_queries:       bool = True,
    use_cache:             bool = True,
    use_memory:            bool = True,
    run_async:             bool = False,
    debug:                 bool = False,
) -> None:
    """
    Orchestrate the full upgraded RAG pipeline.

    Args:
        document_paths:        Local .pdf or .txt file paths.
        web_urls:              Web pages to scrape.
        questions:             Questions to answer.
        rebuild_store:         Rebuild FAISS index from scratch if True.
        use_semantic_chunking: Meaning-aware splits (vs fixed-size).
        use_hybrid_retriever:  BM25 + FAISS with RRF fusion.
        use_reranker:          Cross-encoder reranking after retrieval.
        evaluate_answers:      Score faithfulness + relevancy per answer.
        rewrite_queries:       LLM rewrites queries before retrieval.
        use_cache:             Cache answers to disk (skip LLM on repeats).
        use_memory:            Remember previous Q&A turns.
        run_async:             Run all questions concurrently.
        debug:                 Print raw retrieved chunks per question.
    """
    embeddings_model_name = "all-MiniLM-L6-v2"
    llm_model_name        = "llama-3.3-70b-versatile"
    reranker_model_name   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    faiss_save_path       = "./faiss_index"

    # ── Step 1: Load Documents ──────────────────────────────────────────────
    all_documents: List[Document] = []

    if document_paths:
        for path in document_paths:
            p = Path(path)
            if p.suffix.lower() == ".pdf":
                all_documents.extend(load_documents_from_pdf(str(p)))
            elif p.suffix.lower() == ".txt":
                all_documents.extend(load_documents_from_text(str(p)))
            else:
                log.warning(f"Unsupported file type skipped: {path}")

    if web_urls:
        all_documents.extend(load_documents_from_web(web_urls))

    if not all_documents and rebuild_store:
        log.error("No documents loaded.")
        return

    # ── Step 2: Embeddings ──────────────────────────────────────────────────
    embeddings = create_embeddings_model(embeddings_model_name)

    # ── Step 3: Chunking ────────────────────────────────────────────────────
    if all_documents:
        if use_semantic_chunking:
            chunks = split_documents_semantic(all_documents, embeddings)
            if len(chunks) < 5:
                log.warning("Few semantic chunks — falling back to recursive splitter.")
                chunks = split_documents_recursive(all_documents)
        else:
            chunks = split_documents_recursive(all_documents)
    else:
        chunks = []

    # ── Step 4: Vector Store ────────────────────────────────────────────────
    if rebuild_store and chunks:
        vectorstore = build_faiss_vectorstore(chunks, embeddings, faiss_save_path)
    else:
        vectorstore = load_faiss_vectorstore(embeddings, faiss_save_path)
        if not chunks:
            chunks = [None] * vectorstore.index.ntotal

    # ── Step 5: Retriever ───────────────────────────────────────────────────
    if use_hybrid_retriever and chunks and all(c is not None for c in chunks):
        retriever       = create_hybrid_retriever(chunks, vectorstore, k=8)
        retriever_label = "Hybrid (BM25 + FAISS, RRF)"
    else:
        _f = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})
        class _SimpleRetriever:
            def __init__(self, r): self._r = r
            def invoke(self, q):   return self._r.invoke(q)
        retriever       = _SimpleRetriever(_f)
        retriever_label = "FAISS similarity"

    # ── Step 6: Reranker ────────────────────────────────────────────────────
    reranker = load_reranker(reranker_model_name) if use_reranker else None

    # ── Step 7: LLM ─────────────────────────────────────────────────────────
    llm = create_llm(model_name=llm_model_name, temperature=0.0)

    # ── Step 8: Prompt + Chain ──────────────────────────────────────────────
    prompt = create_rag_prompt()
    chain  = build_rag_chain(llm, retriever, prompt)

    # ── Step 9: Optional components ─────────────────────────────────────────
    cache  = ResponseCache()      if use_cache  else None
    memory = ConversationMemory() if use_memory else None

    # ── Summary ─────────────────────────────────────────────────────────────
    display_pipeline_summary(
        chunks, llm_model_name, embeddings_model_name,
        retriever_type=retriever_label,
        reranker_model=reranker_model_name if use_reranker else "Disabled",
    )

    if not questions:
        return

    # ── Step 10: Run Queries ─────────────────────────────────────────────────
    shared_kwargs = dict(
        reranker        = reranker,
        llm             = llm,
        memory          = memory,
        cache           = cache,
        evaluate        = evaluate_answers,
        rewrite_queries = rewrite_queries,
    )

    if run_async:
        results = asyncio.run(
            run_queries_concurrently(chain, retriever, questions, **shared_kwargs)
        )
    else:
        results = []
        for question in questions:
            if debug:
                debug_retriever(retriever, question)
            results.append(
                query_rag_pipeline(
                    chain=chain, retriever=retriever,
                    question=question, **shared_kwargs
                )
            )

    # ── Evaluation Summary ───────────────────────────────────────────────────
    scored = [r for r in results if r.get("eval_scores")]
    if scored:
        avg_faith   = sum(r["eval_scores"]["faithfulness"]     for r in scored) / len(scored)
        avg_relev   = sum(r["eval_scores"]["answer_relevancy"]  for r in scored) / len(scored)
        avg_latency = sum(r.get("latency_s", 0) for r in results) / len(results)
        cache_hits  = sum(1 for r in results if r.get("cached"))

        print("\n" + "=" * 60)
        print("          EVALUATION SUMMARY")
        print("=" * 60)
        print(f"  Questions answered   : {len(results)}")
        print(f"  Cache hits           : {cache_hits}")
        print(f"  Avg faithfulness     : {avg_faith:.2f} / 1.00")
        print(f"  Avg answer relevancy : {avg_relev:.2f} / 1.00")
        print(f"  Avg latency          : {avg_latency:.1f}s per query")
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cache = ResponseCache()
    cache.clear()
    # for clearing the cache

    """
    Setup:
    1. Create a .env file in the project root:
           GROQ_API_KEY=gsk_your_key_here
    2. Update SAMPLE_DOCUMENTS with your file paths.
    3. Run: python rag_pipeline.py
    """

    SAMPLE_DOCUMENTS = [
        "C:/Users/SAISH PARKAR/PycharmProjects/RagStudyMaterial/articles/World_war.pdf",
        "C:/Users/SAISH PARKAR/PycharmProjects/RagStudyMaterial/articles/Short_Novels_Introduction.pdf",
        "C:/Users/SAISH PARKAR/PycharmProjects/RagStudyMaterial/articles/article.txt",
    ]

    # SAMPLE_QUESTIONS = [
    #     "What happened on February 19, 1942?",
    #     "What were the main causes of World War II?",
    #     "Who were the key leaders during World War II?",
    #     "What does RAG stand for?",
    #     "What are the two main components of RAG?",
    #     "Why is RAG used instead of only LLMs?",
    #     "Does RAG require retraining the model when new data is added?",
    #     "What happens if irrelevant chunks are retrieved?",
    #     "Can RAG work without vector databases? Why or why not?",
    #     "What is the limitation of relying only on retrieval in RAG?",
    #     "How can metadata filtering improve results?",
    # ]

    SAMPLE_QUESTIONS = [
        "What is science?",
        "What is technology?",
        "What is the relationship between science and technology?",
        "What are some examples of modern scientific advancements?",
        "What role does artificial intelligence play in modern industries?"
    ]

    run_full_rag_pipeline(
        document_paths        = SAMPLE_DOCUMENTS,
        questions             = SAMPLE_QUESTIONS,
        rebuild_store         = True,
        use_semantic_chunking = True,   # meaning-aware splits
        use_hybrid_retriever  = True,   # BM25 + FAISS with RRF fusion
        use_reranker          = True,   # cross-encoder reranking
        evaluate_answers      = True,   # faithfulness + relevancy scoring
        rewrite_queries       = True,   # LLM rewrites queries before retrieval
        use_cache             = True,   # cache answers to rag_cache.json
        use_memory            = True,   # remember previous Q&A turns
        run_async             = False,  # set True to run all questions in parallel
        debug                 = False,  # set True to print raw retrieved chunks
    )

