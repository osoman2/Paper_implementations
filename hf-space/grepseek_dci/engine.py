"""
GrepSeek DCI search engine.

Three retrieval modes:
  - Semantic: BM25 (lexical proxy for dense embeddings in this lightweight POC)
  - DCI:      Python regex over raw document text (mirrors grep/find on a file system)
  - Hybrid:   Reciprocal Rank Fusion over semantic + DCI ranked lists
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .corpus import CORPUS, DOC_MAP, Document  # noqa: F401

RRF_K = 60


# ---------------------------------------------------------------------------
# Build BM25 index once at module level
# ---------------------------------------------------------------------------

_tokenized = [doc.text.lower().split() for doc in CORPUS]
_bm25 = BM25Okapi(_tokenized)
_doc_ids = [doc.id for doc in CORPUS]


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------

def semantic_search(query: str, top_k: int = 5) -> list[tuple[str, float]]:
    """BM25-based semantic search (bag-of-words proxy for dense embeddings)."""
    tokens = query.lower().split()
    scores = _bm25.get_scores(tokens).tolist()
    ranked = sorted(zip(_doc_ids, scores), key=lambda x: x[1], reverse=True)
    return [(doc_id, score) for doc_id, score in ranked if score > 0][:top_k]


def dci_search(pattern: str, top_k: int = 5) -> list[tuple[str, float]]:
    """
    Direct Corpus Interaction: regex search over raw document text.
    Mirrors what GrepSeek does with shell grep/find commands.
    Score = number of non-overlapping matches (normalised to [0, 1]).
    Falls back to literal substring search if pattern is invalid regex.
    """
    counts: list[tuple[str, int]] = []
    for doc in CORPUS:
        try:
            matches = re.findall(pattern, doc.text, re.IGNORECASE | re.MULTILINE)
        except re.error:
            matches = [pattern] if pattern.lower() in doc.text.lower() else []
        if matches:
            counts.append((doc.id, len(matches)))

    if not counts:
        return []

    max_count = max(c for _, c in counts)
    normalised = [(doc_id, count / max_count) for doc_id, count in counts]
    return sorted(normalised, key=lambda x: x[1], reverse=True)[:top_k]


def hybrid_search(
    semantic_query: str,
    dci_pattern: str,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion over semantic + DCI ranked lists.
    score(d) = Σ 1 / (k + rank_i(d))
    """
    sem = semantic_search(semantic_query, top_k=len(CORPUS))
    dci_pat = dci_pattern.strip() if dci_pattern.strip() else semantic_query
    dci = dci_search(dci_pat, top_k=len(CORPUS))

    rrf: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(sem, start=1):
        rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
    for rank, (doc_id, _) in enumerate(dci, start=1):
        rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)

    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _highlight(text: str, pattern: str, max_len: int = 280) -> str:
    snippet = text[:max_len].replace("\n", " ")
    if not pattern:
        return snippet
    try:
        snippet = re.sub(
            f"({re.escape(pattern)})",
            r"**\1**",
            snippet,
            flags=re.IGNORECASE,
        )
    except re.error:
        pass
    return snippet


@dataclass
class SearchResult:
    rank: int
    doc_id: str
    title_snippet: str
    body_snippet: str
    score: float
    doc_type: str


def format_results_md(
    results: list[tuple[str, float]],
    highlight_term: str = "",
) -> str:
    if not results:
        return "_No results found._"
    lines = []
    for rank, (doc_id, score) in enumerate(results, start=1):
        doc = DOC_MAP[doc_id]
        dtype = doc.metadata.get("type", "").upper()
        snippet = _highlight(doc.text, highlight_term)
        lines.append(
            f"**{rank}. [{dtype}]** _(score: {score:.4f})_\n\n> {snippet}…\n"
        )
    return "\n---\n".join(lines)
