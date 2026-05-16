"""
Adaptive alpha computation via IDF-based query analysis (vstash approach).

High-IDF queries (rare/technical terms) → higher sparse weight.
Low-IDF queries (common words) → higher dense weight.
Alpha ∈ [0, 1] where 1 = fully sparse, 0 = fully dense.
"""

import math
import numpy as np
from collections import Counter
from typing import Sequence


class QueryAnalyzer:
    def __init__(self, corpus: list[str], min_alpha: float = 0.1, max_alpha: float = 0.9):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self._df: dict[str, int] = {}
        self._N: int = 0
        self._build_df(corpus)

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _build_df(self, corpus: list[str]) -> None:
        self._N = len(corpus)
        for doc in corpus:
            for token in set(self._tokenize(doc)):
                self._df[token] = self._df.get(token, 0) + 1

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        # smoothed IDF — unknown terms get max rarity
        return math.log((self._N + 1) / (df + 1)) + 1

    def compute_alpha(self, query: str) -> float:
        """Return sparse weight alpha for this query."""
        tokens = self._tokenize(query)
        if not tokens:
            return 0.5

        mean_idf = np.mean([self._idf(t) for t in tokens])
        # IDF range is [1, log(N+1)+1]; normalise to [0,1] via sigmoid
        # centre sigmoid around the corpus median IDF
        max_idf = math.log(self._N + 1) + 1
        normalised = mean_idf / max_idf          # ∈ [~0, 1]
        sigmoid_input = (normalised - 0.5) * 8   # steepen the curve
        sigmoid = 1 / (1 + math.exp(-sigmoid_input))

        alpha = self.min_alpha + sigmoid * (self.max_alpha - self.min_alpha)
        return round(float(alpha), 4)

    def query_type(self, alpha: float) -> str:
        if alpha >= 0.65:
            return "keyword-heavy (sparse favoured)"
        if alpha <= 0.35:
            return "semantic (dense favoured)"
        return "balanced"
