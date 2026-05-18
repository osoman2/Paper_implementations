"""
Attribution Crisis — POC scorer.

Measures how much of a RAG-generated response can be traced back to its
retrieved source documents, exposing the "attribution gap" discussed in the paper:
"The attribution crisis in LLM search results: Estimating ecosystem exploitation".

Core idea: when an LLM synthesises an answer from retrieved passages it often
reproduces substantial portions of copyrighted content without citing the source.
This scorer quantifies that gap using:

  - Lexical overlap  (n-gram coverage between response and each source)
  - Semantic overlap (cosine similarity of sentence embeddings)
  - Citation recall  (fraction of cited claims that are actually supported)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Document:
    id: str
    title: str
    text: str
    url: str = ""


@dataclass
class RagResponse:
    query: str
    answer: str
    cited_ids: list[str] = field(default_factory=list)


@dataclass
class SourceAttribution:
    doc_id: str
    doc_title: str
    lexical_overlap: float   # 0-1  fraction of response n-grams found in doc
    semantic_overlap: float  # 0-1  cosine similarity of embeddings
    is_cited: bool


@dataclass
class AttributionReport:
    query: str
    answer: str
    source_attributions: list[SourceAttribution]
    # Aggregates
    mean_lexical_coverage: float    # avg lexical overlap across sources
    max_lexical_coverage: float     # highest overlap with a single source
    mean_semantic_coverage: float
    max_semantic_coverage: float
    citation_recall: float          # fraction of top sources that are cited
    attribution_score: float        # composite 0-1 (higher = better attribution)
    exploitation_risk: str          # LOW / MEDIUM / HIGH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ngrams(text: str, n: int) -> set[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def _lexical_overlap(response: str, document: str, n: int = 3) -> float:
    """Fraction of response n-grams that appear in the document."""
    resp_grams = _ngrams(response, n)
    if not resp_grams:
        return 0.0
    doc_grams = _ngrams(document, n)
    return len(resp_grams & doc_grams) / len(resp_grams)


def _tfidf_vector(text: str, vocab: dict[str, int], idf: np.ndarray) -> np.ndarray:
    tokens = re.findall(r"\b\w+\b", text.lower())
    tf: dict[int, float] = {}
    for tok in tokens:
        idx = vocab.get(tok)
        if idx is not None:
            tf[idx] = tf.get(idx, 0) + 1
    vec = np.zeros(len(vocab))
    for idx, count in tf.items():
        vec[idx] = (count / max(len(tokens), 1)) * idf[idx]
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _build_vocab_idf(docs: list[str]) -> tuple[dict[str, int], np.ndarray]:
    token_sets = [set(re.findall(r"\b\w+\b", d.lower())) for d in docs]
    all_tokens = sorted({t for s in token_sets for t in s})
    vocab = {t: i for i, t in enumerate(all_tokens)}
    N = len(docs)
    idf = np.zeros(len(vocab))
    for i, t in enumerate(all_tokens):
        df = sum(1 for s in token_sets if t in s)
        idf[i] = np.log((N + 1) / (df + 1)) + 1.0
    return vocab, idf


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

class AttributionScorer:
    """
    Scores attribution between a RAG response and its retrieved source documents.

    Uses lightweight TF-IDF cosine similarity (no external model required) plus
    n-gram lexical overlap to estimate how much response content can be traced to
    each source.
    """

    def __init__(self, documents: list[Document]):
        self.documents = {doc.id: doc for doc in documents}
        corpus = [doc.text for doc in documents]
        self._vocab, self._idf = _build_vocab_idf(corpus)
        self._doc_vecs: dict[str, np.ndarray] = {}
        for doc in documents:
            self._doc_vecs[doc.id] = _tfidf_vector(doc.text, self._vocab, self._idf)

    def score(self, response: RagResponse) -> AttributionReport:
        resp_vec = _tfidf_vector(response.answer, self._vocab, self._idf)

        attributions: list[SourceAttribution] = []
        for doc_id, doc in self.documents.items():
            lex = _lexical_overlap(response.answer, doc.text, n=3)
            sem = float(np.dot(resp_vec, self._doc_vecs[doc_id]))
            attributions.append(SourceAttribution(
                doc_id=doc_id,
                doc_title=doc.title,
                lexical_overlap=round(lex, 4),
                semantic_overlap=round(sem, 4),
                is_cited=doc_id in response.cited_ids,
            ))

        attributions.sort(key=lambda a: a.semantic_overlap, reverse=True)

        lex_vals = [a.lexical_overlap for a in attributions]
        sem_vals = [a.semantic_overlap for a in attributions]

        mean_lex = round(float(np.mean(lex_vals)), 4)
        max_lex  = round(float(np.max(lex_vals)), 4)
        mean_sem = round(float(np.mean(sem_vals)), 4)
        max_sem  = round(float(np.max(sem_vals)), 4)

        # Citation recall: of top-3 sources by semantic overlap, how many are cited?
        top3 = [a.doc_id for a in attributions[:3]]
        cited_top3 = sum(1 for d in top3 if d in response.cited_ids)
        cit_recall = round(cited_top3 / max(len(top3), 1), 4)

        # Composite attribution score (higher = more transparent attribution)
        # Penalises high exploitation with low citation
        exploit_signal = max_lex * (1 - cit_recall)
        attr_score = round(max(0.0, min(1.0, cit_recall * 0.5 + (1 - exploit_signal) * 0.5)), 4)

        # Exploitation risk label
        if exploit_signal > 0.25:
            risk = "HIGH"
        elif exploit_signal > 0.10:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return AttributionReport(
            query=response.query,
            answer=response.answer,
            source_attributions=attributions,
            mean_lexical_coverage=mean_lex,
            max_lexical_coverage=max_lex,
            mean_semantic_coverage=mean_sem,
            max_semantic_coverage=max_sem,
            citation_recall=cit_recall,
            attribution_score=attr_score,
            exploitation_risk=risk,
        )
