"""
Sparse (BM25) and dense (sentence-transformer) retrievers.
Both return (doc_id, score) lists sorted descending.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


@dataclass
class Document:
    id: str
    text: str
    metadata: dict


class SparseRetriever:
    def __init__(self, documents: list[Document]):
        self._docs = documents
        tokenized = [doc.text.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip([d.id for d in self._docs], scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class DenseRetriever:
    def __init__(self, documents: list[Document], model_name: str = "all-MiniLM-L6-v2"):
        self._docs = documents
        self._model = SentenceTransformer(model_name)
        texts = [doc.text for doc in documents]
        self._embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        q_emb = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        # cosine similarity via dot product on normalised vectors
        sims = (self._embeddings @ q_emb.T).squeeze()
        scores = sims.tolist() if hasattr(sims, "tolist") else list(sims)
        ranked = sorted(zip([d.id for d in self._docs], scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
