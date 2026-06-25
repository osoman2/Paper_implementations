"""
GrepSeek DCI demo corpus.

Designed to surface the tradeoff between semantic search (good at natural language)
and DCI/grep (good at exact strings: error codes, file paths, version constraints).
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

CORPUS = [
    Document("doc1", (
        "LLM inference pipeline crashed with CUDA out-of-memory error OOM_CODE_0x4F5E. "
        "The GPU failed to allocate 6.4 GB for the KV cache. "
        "Stack trace: inference_engine.py:312 RuntimeError: CUDA OOM_CODE_0x4F5E allocate 6.4GB."
    ), {"type": "log"}),

    Document("doc2", (
        "RAG pipeline configuration: embedding_model=all-MiniLM-L6-v2 "
        "chunk_size=512 chunk_overlap=64 top_k=5 "
        "vector_db=chromadb>=0.4.22 reranker=cross-encoder/ms-marco-MiniLM-L-6-v2 "
        "alpha=0.7  # BM25 weight in hybrid fusion"
    ), {"type": "config"}),

    Document("doc3", (
        "HTTP 429 Too Many Requests — rate limit exceeded for OpenAI API client_id=C-7742. "
        "Retry-After: 60s. Daily token quota: 2,000,000 / 2,000,000 used. "
        "ALERT: RATE_LIMIT_EXCEEDED_C7742 triggered PagerDuty escalation."
    ), {"type": "log"}),

    Document("doc4", (
        "Dense retrieval uses sentence embeddings to find semantically similar documents. "
        "A pre-trained transformer encodes the query and all corpus documents into a shared "
        "vector space. Cosine similarity determines relevance. Works well for paraphrase "
        "detection, semantic search, and knowledge-base Q&A over natural language."
    ), {"type": "documentation"}),

    Document("doc5", (
        "BM25 Okapi is a bag-of-words ranking function. Given a query, it scores documents "
        "by term frequency and inverse document frequency with length normalisation. "
        "Unlike dense retrieval, BM25 handles exact keyword matching naturally. "
        "Best for technical queries with specific terminology, IDs, or version strings."
    ), {"type": "documentation"}),

    Document("doc6", (
        "requirements.txt snapshot (2026-06-25):\n"
        "transformers>=4.36.0,<5.0\n"
        "sentence-transformers==3.0.1\n"
        "rank-bm25>=0.2.2\n"
        "chromadb>=0.4.22\n"
        "anthropic>=0.30.0\n"
        "torch==2.2.1+cu121\n"
        "faiss-gpu==1.7.4"
    ), {"type": "config"}),

    Document("doc7", (
        "Direct Corpus Interaction (DCI) bypasses vector embeddings entirely. "
        "An agent issues shell commands — grep, find, awk, sed — directly against "
        "raw files. This enables precise lookup of error codes, version constraints, "
        "and file paths that embeddings would blur or miss. GrepSeek extends DCI "
        "with an RL-trained model that learns to navigate file systems efficiently."
    ), {"type": "documentation"}),

    Document("doc8", (
        "Incident ALERT-INC-2892 (2026-06-24 09:20 UTC): "
        "vector_db latency spike — ChromaDB responded after 5,001ms (SLA: 200ms). "
        "Fallback triggered: BM25 keyword search returned 8 results in 12ms. "
        "Root cause: index rebuild blocked by OOM_CODE_0x4F5E on the embedding worker. "
        "Postmortem action: add circuit-breaker for vector_db timeouts."
    ), {"type": "log"}),

    Document("doc9", (
        "Hybrid search architecture combines sparse retrieval (BM25/TF-IDF) and dense "
        "retrieval (sentence transformers) using Reciprocal Rank Fusion. "
        "Formula: score(d) = Σ 1 / (k + rank_i(d)), k=60. "
        "Empirically outperforms each component alone: +18% nDCG@10 vs pure semantic "
        "and +12% vs pure BM25 on the BEIR benchmark."
    ), {"type": "documentation"}),

    Document("doc10", (
        "Model deployment config (production):\n"
        "endpoint: https://api.internal/v2/embed\n"
        "model_id: text-embedding-3-small\n"
        "version_constraint: api_version>=2024-01-01\n"
        "timeout_ms: 500\n"
        "max_retries: 3\n"
        "fallback_model: all-MiniLM-L6-v2  # local, no API call\n"
        "ALERT_WEBHOOK: https://hooks.slack.com/T0K8/B04/xxx"
    ), {"type": "config"}),

    Document("doc11", (
        "Retrieval quality degrades when documents contain entity identifiers that "
        "embeddings treat as unknown tokens. Error codes like OOM_CODE_0x4F5E, "
        "version strings like transformers==4.36.0, and alert IDs like ALERT-INC-2892 "
        "lose their identity in the embedding space. A hybrid approach using "
        "DCI grep for exact matching followed by semantic reranking recovers this."
    ), {"type": "documentation"}),

    Document("doc12", (
        "GrepSeek benchmark results (EnterpriseSearch-Bench 2025):\n"
        "Pure semantic RAG: nDCG@10 = 0.712\n"
        "Pure BM25 keyword: nDCG@10 = 0.763\n"
        "GrepSeek DCI only: nDCG@10 = 0.781\n"
        "GrepSeek Hybrid (DCI + semantic + RRF): nDCG@10 = 0.847\n"
        "Throughput with sharded-parallel exec: 7.6× vs sequential grep\n"
        "Source: GrepSeek paper, AlphaSignal deep-dive 2026-05-31"
    ), {"type": "documentation"}),
]

DOC_MAP = {doc.id: doc for doc in CORPUS}
