"""
Semi-supervised pseudo-labeling via k-NN in TF-IDF embedding space.

Workflow:
  1. Encode all items (labeled + unlabeled) with TF-IDF.
  2. For each unlabeled item, compute cosine similarity to every labeled item.
  3. Assign pseudo-label = majority vote over top-k nearest labeled neighbors.
  4. Only keep pseudo-labels with confidence >= threshold.
  5. Add accepted ones to the labeled pool and repeat.
"""

from __future__ import annotations
import math
import re
from dataclasses import dataclass
from collections import Counter


@dataclass
class LabeledExample:
    text: str
    label: str


@dataclass
class PseudoLabelResult:
    text: str
    predicted_label: str
    confidence: float
    nearest_neighbor_text: str
    accepted: bool


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _build_idf(corpus: list[str]) -> dict[str, float]:
    n = len(corpus)
    df: dict[str, int] = {}
    for text in corpus:
        for w in set(_tokenize(text)):
            df[w] = df.get(w, 0) + 1
    return {w: math.log((n + 1) / (cnt + 1)) + 1 for w, cnt in df.items()}


def _tfidf_vec(text: str, idf: dict[str, float]) -> dict[str, float]:
    tokens = _tokenize(text)
    if not tokens:
        return {}
    tf = Counter(tokens)
    vec = {w: (c / len(tokens)) * idf[w] for w, c in tf.items() if w in idf}
    norm = math.sqrt(sum(v ** 2 for v in vec.values())) or 1.0
    return {w: v / norm for w, v in vec.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    return sum(a[k] * b[k] for k in set(a) & set(b))


def _predict(
    text: str,
    pool: list[LabeledExample],
    idf: dict[str, float],
    k: int,
    threshold: float,
) -> PseudoLabelResult:
    uvec = _tfidf_vec(text, idf)
    labeled_vecs = [_tfidf_vec(e.text, idf) for e in pool]
    sims = sorted(
        [(_cosine(uvec, lv), e) for lv, e in zip(labeled_vecs, pool)],
        key=lambda x: -x[0],
    )
    top_k = sims[:k]
    if not top_k or top_k[0][0] == 0:
        return PseudoLabelResult(
            text=text, predicted_label="unknown", confidence=0.0,
            nearest_neighbor_text=pool[0].text if pool else "", accepted=False,
        )

    vote: dict[str, float] = {}
    for sim, ex in top_k:
        vote[ex.label] = vote.get(ex.label, 0.0) + sim
    best_label = max(vote, key=vote.__getitem__)
    top_sim = top_k[0][0]
    confidence = round(vote[best_label] / (sum(vote.values()) or 1.0) * top_sim, 4)

    return PseudoLabelResult(
        text=text,
        predicted_label=best_label,
        confidence=confidence,
        nearest_neighbor_text=top_k[0][1].text,
        accepted=confidence >= threshold,
    )


class SemiSupervisedLabeler:
    """
    Iterative pseudo-label expansion using k-NN in TF-IDF space.

    Parameters
    ----------
    confidence_threshold : float
        Minimum cosine similarity * vote-ratio to accept a pseudo-label.
    max_rounds : int
        Maximum self-training rounds.
    k : int
        Nearest labeled neighbors for majority vote.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        max_rounds: int = 3,
        k: int = 3,
    ):
        self.threshold = confidence_threshold
        self.max_rounds = max_rounds
        self.k = k

    def fit_predict(
        self,
        labeled: list[LabeledExample],
        unlabeled: list[str],
    ) -> tuple[list[PseudoLabelResult], list[dict]]:
        """Run iterative pseudo-labeling and return final results + round stats."""
        pool = list(labeled)
        remaining = list(unlabeled)
        # final_decisions stores the result per unlabeled text (updated each round)
        final_decisions: dict[str, PseudoLabelResult] = {}
        round_stats: list[dict] = []

        for rnd in range(self.max_rounds):
            if not remaining:
                break

            idf = _build_idf([e.text for e in pool] + remaining)
            newly_accepted: list[str] = []
            still_remaining: list[str] = []

            for utext in remaining:
                result = _predict(utext, pool, idf, self.k, self.threshold)
                final_decisions[utext] = result
                if result.accepted:
                    pool.append(LabeledExample(text=utext, label=result.predicted_label))
                    newly_accepted.append(utext)
                else:
                    still_remaining.append(utext)

            pseudo_dist = Counter(
                final_decisions[t].predicted_label
                for t in unlabeled
                if t in final_decisions and final_decisions[t].accepted
            )
            round_stats.append({
                "round": rnd + 1,
                "accepted": len(newly_accepted),
                "remaining": len(still_remaining),
                "pool_size": len(pool),
                "label_distribution": dict(pseudo_dist),
            })
            remaining = still_remaining

        # For any still-remaining items, do a final prediction without threshold filtering
        if remaining:
            idf = _build_idf([e.text for e in pool] + remaining)
            for utext in remaining:
                result = _predict(utext, pool, idf, self.k, threshold=0.0)
                result.accepted = False
                final_decisions[utext] = result

        return [final_decisions.get(t, PseudoLabelResult(
            text=t, predicted_label="unknown", confidence=0.0,
            nearest_neighbor_text="", accepted=False,
        )) for t in unlabeled], round_stats
