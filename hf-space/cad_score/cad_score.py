"""
CAD Score — Context-Aware Semantic Diversity.

Paper: "CAD score: context-aware semantic diversity for scalable creativity
evaluation of LLMs" — Kim et al., Konkuk University (PeerJ Computer Science, 2025).

Problem: existing diversity metrics ignore the prompt context, so they measure
raw output variation rather than whether outputs are *meaningfully* diverse
relative to what was asked.

CAD score computes pairwise semantic distances between outputs, then discounts
similarity that merely echoes the prompt — penalising lazy variation that just
rephrases the question rather than exploring different ideas.

Formula (simplified):
    CAD(responses, prompt) = mean pairwise_distance(r_i, r_j) × context_penalty
where context_penalty = 1 - mean sim(r_i, prompt) over all i

Three conventional baselines are also computed for comparison:
  - Distinct-N: lexical diversity (n-gram overlap)
  - Embedding cosine spread: mean pairwise cosine distance (no context adjustment)
  - Self-BLEU: inverse lexical overlap (lower = more diverse)
"""

from __future__ import annotations

import itertools
import math
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer


_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# ---------------------------------------------------------------------------
# CAD score
# ---------------------------------------------------------------------------

def cad_score(prompt: str, responses: list[str]) -> dict:
    """
    Returns CAD score and all intermediate values.

    Args:
        prompt: the original prompt given to the LLM
        responses: list of outputs to evaluate (≥2 required)

    Returns dict with keys:
        cad, context_penalty, raw_spread,
        pairwise_distances, prompt_similarities
    """
    if len(responses) < 2:
        raise ValueError("Need at least 2 responses to compute diversity.")

    model = _get_model()
    texts = [prompt] + responses
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    prompt_emb = embeddings[0]
    resp_embs  = embeddings[1:]

    # pairwise cosine distances between responses (1 - cosine_sim for L2-normed vecs)
    pairs = list(itertools.combinations(range(len(responses)), 2))
    pairwise_dists = [
        float(1 - np.dot(resp_embs[i], resp_embs[j]))
        for i, j in pairs
    ]
    raw_spread = float(np.mean(pairwise_dists))

    # similarity of each response to the prompt
    prompt_sims = [float(np.dot(prompt_emb, r)) for r in resp_embs]
    mean_prompt_sim = float(np.mean(prompt_sims))

    # context penalty: responses highly similar to prompt = low penalty (they're echoing)
    context_penalty = 1.0 - mean_prompt_sim

    cad = raw_spread * context_penalty

    return {
        "cad": round(cad, 4),
        "raw_spread": round(raw_spread, 4),
        "context_penalty": round(context_penalty, 4),
        "mean_prompt_similarity": round(mean_prompt_sim, 4),
        "pairwise_distances": [round(d, 4) for d in pairwise_dists],
        "prompt_similarities": [round(s, 4) for s in prompt_sims],
    }


# ---------------------------------------------------------------------------
# Baseline metrics
# ---------------------------------------------------------------------------

def _ngrams(text: str, n: int) -> Counter:
    tokens = text.lower().split()
    return Counter(zip(*[tokens[i:] for i in range(n)]))


def distinct_n(responses: list[str], n: int = 2) -> float:
    """Distinct-N: ratio of unique n-grams to total n-grams across all responses."""
    all_ngrams: list = []
    unique: set = set()
    for r in responses:
        grams = list(_ngrams(r, n).keys())
        all_ngrams.extend(grams)
        unique.update(grams)
    if not all_ngrams:
        return 0.0
    return round(len(unique) / len(all_ngrams), 4)


def embedding_spread(responses: list[str]) -> float:
    """Mean pairwise cosine distance between response embeddings (no context adjustment)."""
    model = _get_model()
    embs = model.encode(responses, normalize_embeddings=True, show_progress_bar=False)
    pairs = list(itertools.combinations(range(len(responses)), 2))
    dists = [float(1 - np.dot(embs[i], embs[j])) for i, j in pairs]
    return round(float(np.mean(dists)), 4) if dists else 0.0


def self_bleu(responses: list[str], n: int = 2) -> float:
    """
    Self-BLEU: average n-gram overlap of each response against all others.
    Lower = more diverse.
    """
    scores = []
    for i, ref in enumerate(responses):
        others = [responses[j] for j in range(len(responses)) if j != i]
        ref_grams = _ngrams(ref, n)
        if not ref_grams:
            continue
        overlap = 0
        for other in others:
            other_grams = _ngrams(other, n)
            overlap += sum((ref_grams & other_grams).values())
        precision = overlap / (sum(ref_grams.values()) * len(others)) if others else 0
        scores.append(precision)
    return round(float(np.mean(scores)), 4) if scores else 0.0


def all_metrics(prompt: str, responses: list[str]) -> dict:
    """Compute CAD + all baselines in one call."""
    cad = cad_score(prompt, responses)
    return {
        **cad,
        "distinct_2": distinct_n(responses, 2),
        "embedding_spread": embedding_spread(responses),
        "self_bleu_2": self_bleu(responses, 2),
    }
