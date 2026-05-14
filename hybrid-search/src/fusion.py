"""
Result fusion strategies for hybrid search.

- score_fusion: min-max normalise each list then blend with alpha
- rrf_fusion: Reciprocal Rank Fusion (rank-only, no score normalisation needed)
Both accept alpha ∈ [0,1]: 1 = fully sparse, 0 = fully dense.
"""

from __future__ import annotations


def _min_max(scores: list[float]) -> list[float]:
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span == 0:
        return [1.0] * len(scores)
    return [(s - lo) / span for s in scores]


def score_fusion(
    sparse_results: list[tuple[str, float]],
    dense_results: list[tuple[str, float]],
    alpha: float,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Weighted combination after min-max normalisation."""
    s_ids, s_scores = zip(*sparse_results) if sparse_results else ([], [])
    d_ids, d_scores = zip(*dense_results) if dense_results else ([], [])

    s_norm = dict(zip(s_ids, _min_max(list(s_scores)))) if s_scores else {}
    d_norm = dict(zip(d_ids, _min_max(list(d_scores)))) if d_scores else {}

    all_ids = set(s_norm) | set(d_norm)
    fused: dict[str, float] = {}
    for doc_id in all_ids:
        fused[doc_id] = alpha * s_norm.get(doc_id, 0.0) + (1 - alpha) * d_norm.get(doc_id, 0.0)

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def rrf_fusion(
    sparse_results: list[tuple[str, float]],
    dense_results: list[tuple[str, float]],
    alpha: float,
    top_k: int = 5,
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion with alpha-weighted contribution per list.
    alpha weights sparse list; (1-alpha) weights dense list.
    """
    fused: dict[str, float] = {}

    for rank, (doc_id, _) in enumerate(sparse_results):
        fused[doc_id] = fused.get(doc_id, 0.0) + alpha * (1 / (k + rank + 1))

    for rank, (doc_id, _) in enumerate(dense_results):
        fused[doc_id] = fused.get(doc_id, 0.0) + (1 - alpha) * (1 / (k + rank + 1))

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
