"""
Bi-Stage Adaptive RAG — POC based on:
'Group-Based Recommendation System Using Bi-Stage Adaptive Deep Learning Model' (score 8/10)

Key idea: two-stage pipeline
  Stage 1 — Candidate Generation: lightweight BM25 per query aspect → merged candidate pool
  Stage 2 — Adaptive Reranking: attention weights over aspects, dynamically computed from
             aspect-query similarity, re-score candidates without any offline training
"""

from .retriever import MultiAspectQuery, Stage1Retriever
from .reranker import AdaptiveReranker, RerankResult

__all__ = ["MultiAspectQuery", "Stage1Retriever", "AdaptiveReranker", "RerankResult"]
