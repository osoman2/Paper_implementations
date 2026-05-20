"""Stage 2 — Adaptive attention reranker.

Computes per-aspect attention weights from aspect-query cosine similarity
(using TF-IDF bag-of-words embeddings), then re-scores each candidate as
a weighted sum of its per-aspect BM25 scores.  No offline training needed.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class RerankResult:
    rank: int
    doc_id: str
    final_score: float
    aspect_weights: dict[str, float]
    aspect_scores: dict[str, float]
    text: str = ""


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _tfidf_vec(tokens: list[str], vocab: dict[str, int], idf: dict[str, float]) -> list[float]:
    tf = Counter(tokens)
    total = max(sum(tf.values()), 1)
    vec = [0.0] * len(vocab)
    for t, idx in vocab.items():
        vec[idx] = (tf.get(t, 0) / total) * idf.get(t, 0.0)
    return vec


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


def _softmax(vals: list[float], temperature: float = 1.0) -> list[float]:
    scaled = [v / temperature for v in vals]
    m = max(scaled)
    exps = [math.exp(v - m) for v in scaled]
    s = sum(exps)
    return [e / s for e in exps]


class AdaptiveReranker:
    """Computes attention weights over aspects via TF-IDF cosine similarity."""

    def rerank(
        self,
        candidates: list,
        query_text: str,
        aspect_texts: list[str],
        doc_map: dict[str, "Document"],
        temperature: float = 0.5,
        fixed_weights: list[float] | None = None,
    ) -> list[RerankResult]:
        all_texts = [doc_map[c.doc_id].text for c in candidates]
        all_texts += [query_text] + aspect_texts

        # Build vocab and IDF from local mini-corpus
        all_token_lists = [_tokenize(t) for t in all_texts]
        N = len(all_token_lists)
        df: Counter = Counter()
        for toks in all_token_lists:
            for t in set(toks):
                df[t] += 1
        idf = {t: math.log((N + 1) / (n + 1)) for t, n in df.items()}
        vocab = {t: i for i, t in enumerate(sorted(idf.keys()))}

        query_vec = _tfidf_vec(_tokenize(query_text), vocab, idf)
        aspect_vecs = [_tfidf_vec(_tokenize(a), vocab, idf) for a in aspect_texts]

        # Attention: how similar is each aspect to the main query?
        sims = [_cosine(query_vec, av) for av in aspect_vecs]
        if fixed_weights is not None:
            attn = fixed_weights
        else:
            attn = _softmax(sims, temperature=temperature)

        aspect_names = ["main_query"] + [f"aspect_{i+1}" for i in range(len(aspect_texts))]

        results = []
        for cand in candidates:
            # Build final score as weighted sum over aspect BM25 scores
            aspect_score_map = cand.aspect_scores
            final_score = 0.0
            for asp_name, w in zip(aspect_names, [1.0] + attn):
                sc = aspect_score_map.get(asp_name, 0.0)
                # main_query has fixed weight 1.0, aspects use attention weights
                # blend: 0.5 * main + 0.5 * sum(w_i * aspect_i)
                if asp_name == "main_query":
                    final_score += 0.5 * sc
                else:
                    final_score += 0.5 * w * sc

            weights_display = {"main_query (fixed 0.5)": 0.5}
            for i, (an, w) in enumerate(zip(aspect_names[1:], attn)):
                weights_display[an] = round(w * 0.5, 4)

            results.append(
                RerankResult(
                    rank=0,
                    doc_id=cand.doc_id,
                    final_score=final_score,
                    aspect_weights=weights_display,
                    aspect_scores=dict(aspect_score_map),
                    text=doc_map[cand.doc_id].text,
                )
            )

        results.sort(key=lambda r: r.final_score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1
        return results
