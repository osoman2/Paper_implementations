"""Stage 1 — multi-aspect candidate generation using BM25."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Document:
    id: str
    text: str


@dataclass
class MultiAspectQuery:
    text: str
    aspects: list[str]

    def all_terms(self) -> list[str]:
        combined = self.text + " " + " ".join(self.aspects)
        return re.findall(r"\w+", combined.lower())


@dataclass
class CandidateScore:
    doc_id: str
    aspect_scores: dict[str, float] = field(default_factory=dict)

    @property
    def max_score(self) -> float:
        return max(self.aspect_scores.values(), default=0.0)


class Stage1Retriever:
    """BM25 retriever that scores each document against each query aspect."""

    def __init__(self, corpus: list[Document], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _build_index(self) -> None:
        self.doc_tokens: dict[str, list[str]] = {
            d.id: self._tokenize(d.text) for d in self.corpus
        }
        self.doc_len = {did: len(toks) for did, toks in self.doc_tokens.items()}
        self.avgdl = sum(self.doc_len.values()) / max(len(self.doc_len), 1)
        N = len(self.corpus)

        # IDF per term
        df: Counter = Counter()
        for toks in self.doc_tokens.values():
            for t in set(toks):
                df[t] += 1
        self.idf: dict[str, float] = {
            t: math.log((N - n + 0.5) / (n + 0.5) + 1.0)
            for t, n in df.items()
        }

    def _bm25_score(self, query_terms: list[str], doc_id: str) -> float:
        toks = self.doc_tokens[doc_id]
        tf = Counter(toks)
        dl = self.doc_len[doc_id]
        score = 0.0
        for t in query_terms:
            if t not in self.idf:
                continue
            f = tf.get(t, 0)
            num = f * (self.k1 + 1)
            den = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf[t] * num / den
        return score

    def generate_candidates(
        self, query: MultiAspectQuery, top_k_per_aspect: int = 10
    ) -> list[CandidateScore]:
        """Score every document against each aspect; return union top-K pool."""
        all_aspects = [query.text] + query.aspects
        aspect_names = ["main_query"] + [f"aspect_{i+1}" for i in range(len(query.aspects))]

        scores: dict[str, CandidateScore] = {}
        for asp_name, asp_text in zip(aspect_names, all_aspects):
            terms = self._tokenize(asp_text)
            doc_scores = [
                (d.id, self._bm25_score(terms, d.id)) for d in self.corpus
            ]
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            for doc_id, sc in doc_scores[:top_k_per_aspect]:
                if doc_id not in scores:
                    scores[doc_id] = CandidateScore(doc_id=doc_id)
                scores[doc_id].aspect_scores[asp_name] = sc

        return list(scores.values())
