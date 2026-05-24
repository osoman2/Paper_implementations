"""
Extreme Event AI — Reliability Calibration

Implements the paper's core insight applied to RAG/retrieval:

1. ExtremeQueryClassifier: classifies queries as "normal" or "extreme" based on
   distributional features (IDF rarity, length outliers, vocabulary novelty).
   Analogous to classifying weather events as normal vs. extreme precipitation.

2. ReliabilityCalibrator: computes Expected Calibration Error (ECE) and produces
   reliability diagram data — answers "when this model says 80% confident, is it
   right ~80% of the time?". Mirrors the paper's reliability diagrams for AI
   weather models.

3. ModelComparison: runs multiple strategies (BM25-only, dense-only, hybrid, reranker)
   and compares their reliability on normal vs. extreme queries. The paper's central
   finding: AI models that perform well on average can be systematically unreliable
   on extremes. Same is true for RAG retrievers.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """Single prediction result from a retrieval/prediction strategy."""
    model_name: str
    query: str
    query_type: str          # "normal" or "extreme"
    predicted_score: float   # model's confidence 0-1 (e.g., retrieval relevance score)
    actual_correct: bool     # was the prediction correct (ground truth)?
    percentile: float        # query's extremity percentile (0=normal, 1=most extreme)


@dataclass
class CalibrationReport:
    """Full calibration report for one model across normal+extreme splits."""
    model_name: str
    # Per-bucket reliability diagram data: (mean_confidence, fraction_correct, n_samples)
    reliability_buckets_normal: list[tuple[float, float, int]]
    reliability_buckets_extreme: list[tuple[float, float, int]]
    # Summary metrics
    ece_normal: float          # Expected Calibration Error on normal queries
    ece_extreme: float         # ECE on extreme queries (usually higher = less reliable)
    accuracy_normal: float
    accuracy_extreme: float
    overconfidence_normal: float   # mean(predicted_score) - accuracy; + = overconfident
    overconfidence_extreme: float
    n_normal: int
    n_extreme: int


# ---------------------------------------------------------------------------
# Extreme query classifier
# ---------------------------------------------------------------------------

class ExtremeQueryClassifier:
    """
    Classifies queries as normal vs. extreme based on vocabulary rarity and
    distributional features.

    Analogous to the paper's distinction between normal precipitation events
    and extreme ones (top 95th percentile).

    Features used:
      - Mean IDF of query tokens (rare tokens → higher IDF → more "extreme")
      - Query length deviation from corpus mean
      - OOV (out-of-vocabulary) fraction
    """

    def __init__(self, corpus: list[str], extreme_percentile: float = 0.85):
        self.extreme_percentile = extreme_percentile
        self._build_vocab(corpus)

    def _build_vocab(self, corpus: list[str]):
        import re
        # Build IDF over corpus
        token_sets = []
        for doc in corpus:
            tokens = set(re.findall(r'\b\w+\b', doc.lower()))
            token_sets.append(tokens)

        all_tokens = {t for s in token_sets for t in s}
        N = len(corpus) or 1
        self._idf: dict[str, float] = {}
        for tok in all_tokens:
            df = sum(1 for s in token_sets if tok in s)
            self._idf[tok] = math.log((N + 1) / (df + 1)) + 1.0

        # Corpus token statistics
        all_lengths = [len(s) for s in token_sets]
        self._mean_len = sum(all_lengths) / max(len(all_lengths), 1)
        self._std_len = (
            math.sqrt(sum((l - self._mean_len) ** 2 for l in all_lengths) / max(len(all_lengths), 1))
            or 1.0
        )
        self._vocab = all_tokens
        self._mean_idf = (
            sum(self._idf.values()) / max(len(self._idf), 1)
        )

    def extremity_score(self, query: str) -> float:
        """Returns 0–1 extremity score: higher = more extreme/unusual query."""
        import re
        tokens = re.findall(r'\b\w+\b', query.lower())
        if not tokens:
            return 0.0

        # Mean IDF of query tokens (unknown tokens get max IDF proxy)
        max_idf = max(self._idf.values()) if self._idf else 5.0
        mean_q_idf = sum(self._idf.get(t, max_idf) for t in tokens) / len(tokens)
        idf_factor = min(mean_q_idf / (self._mean_idf + 1e-9), 3.0) / 3.0

        # Length deviation
        len_z = abs(len(tokens) - self._mean_len) / self._std_len
        len_factor = min(len_z / 3.0, 1.0)

        # OOV fraction
        oov_frac = sum(1 for t in tokens if t not in self._vocab) / len(tokens)

        # Weighted combination
        score = 0.5 * idf_factor + 0.3 * oov_frac + 0.2 * len_factor
        return round(min(score, 1.0), 4)

    def classify(self, queries: list[str]) -> list[tuple[str, float, str]]:
        """
        Returns list of (query, extremity_score, type) where type ∈ {normal, extreme}.
        Uses the extreme_percentile threshold.
        """
        scored = [(q, self.extremity_score(q)) for q in queries]
        scores = [s for _, s in scored]
        if not scores:
            return []
        scores_sorted = sorted(scores)
        threshold_idx = int(len(scores_sorted) * self.extreme_percentile)
        threshold = scores_sorted[min(threshold_idx, len(scores_sorted) - 1)]

        return [
            (q, score, "extreme" if score >= threshold else "normal")
            for q, score in scored
        ]


# ---------------------------------------------------------------------------
# Simulated retrieval model predictions
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class _SimulatedModel:
    """
    Simulates a retrieval/prediction model with known strengths and weaknesses.

    Each model has:
      - base_accuracy: overall accuracy on normal queries
      - extreme_degradation: how much accuracy drops on extreme queries
      - overconfidence_bias: tendency to report high confidence even when wrong
    """

    def __init__(
        self,
        name: str,
        base_accuracy: float,
        extreme_degradation: float,
        overconfidence_bias: float,
        seed: int = 42,
    ):
        self.name = name
        self.base_accuracy = base_accuracy
        self.extreme_degradation = extreme_degradation
        self.overconfidence_bias = overconfidence_bias
        self._rng = random.Random(seed)

    def predict(self, query: str, extremity: float) -> tuple[float, bool]:
        """Returns (confidence, is_correct)."""
        # True accuracy degrades for extreme queries
        true_accuracy = max(
            0.05,
            self.base_accuracy - self.extreme_degradation * extremity,
        )
        is_correct = self._rng.random() < true_accuracy

        # Confidence can be inflated by overconfidence_bias
        # On correct predictions: confidence ≈ true_accuracy + bias
        # On incorrect predictions: confidence ≈ 1 - true_accuracy + bias (overconfident when wrong)
        if is_correct:
            raw_conf = self._rng.gauss(true_accuracy + self.overconfidence_bias * 0.3, 0.1)
        else:
            raw_conf = self._rng.gauss(
                true_accuracy + self.overconfidence_bias * extremity * 0.5, 0.12
            )
        confidence = max(0.05, min(0.99, raw_conf))
        return round(confidence, 4), is_correct


# Pre-defined models representing common RAG strategies
BUILTIN_MODELS = {
    "BM25 (sparse)": _SimulatedModel(
        "BM25 (sparse)",
        base_accuracy=0.78,
        extreme_degradation=0.30,  # BM25 struggles with rare/novel queries
        overconfidence_bias=0.05,
        seed=1,
    ),
    "Dense retrieval": _SimulatedModel(
        "Dense retrieval",
        base_accuracy=0.82,
        extreme_degradation=0.20,  # Dense generalises better
        overconfidence_bias=0.12,  # Embedding models can be overconfident
        seed=2,
    ),
    "Hybrid (fixed α=0.5)": _SimulatedModel(
        "Hybrid (fixed α=0.5)",
        base_accuracy=0.84,
        extreme_degradation=0.22,
        overconfidence_bias=0.08,
        seed=3,
    ),
    "Adaptive Hybrid (query-IDF α)": _SimulatedModel(
        "Adaptive Hybrid (query-IDF α)",
        base_accuracy=0.86,
        extreme_degradation=0.12,  # Adapts to query type → better on extremes
        overconfidence_bias=0.06,
        seed=4,
    ),
    "Cross-encoder reranker": _SimulatedModel(
        "Cross-encoder reranker",
        base_accuracy=0.88,
        extreme_degradation=0.15,
        overconfidence_bias=0.04,
        seed=5,
    ),
}


# ---------------------------------------------------------------------------
# Reliability calibrator
# ---------------------------------------------------------------------------

class ReliabilityCalibrator:
    """
    Computes Expected Calibration Error (ECE) and reliability diagram data.

    The paper evaluates AI weather models using reliability diagrams that ask:
    when the model assigns probability p to an event, does that event occur p of
    the time? Perfect calibration lies on the y=x diagonal.

    Applied to RAG: when the retriever scores a document 0.8, does it turn out
    to be the correct top result ~80% of the time?
    """

    def __init__(self, n_buckets: int = 10):
        self.n_buckets = n_buckets

    def compute(self, results: list[ModelResult]) -> CalibrationReport:
        normal  = [r for r in results if r.query_type == "normal"]
        extreme = [r for r in results if r.query_type == "extreme"]

        rel_normal  = self._reliability_buckets(normal)
        rel_extreme = self._reliability_buckets(extreme)

        ece_normal   = self._ece(normal)
        ece_extreme  = self._ece(extreme)
        acc_normal   = self._accuracy(normal)
        acc_extreme  = self._accuracy(extreme)
        oc_normal    = self._overconfidence(normal)
        oc_extreme   = self._overconfidence(extreme)

        return CalibrationReport(
            model_name=results[0].model_name if results else "?",
            reliability_buckets_normal=rel_normal,
            reliability_buckets_extreme=rel_extreme,
            ece_normal=round(ece_normal, 4),
            ece_extreme=round(ece_extreme, 4),
            accuracy_normal=round(acc_normal, 4),
            accuracy_extreme=round(acc_extreme, 4),
            overconfidence_normal=round(oc_normal, 4),
            overconfidence_extreme=round(oc_extreme, 4),
            n_normal=len(normal),
            n_extreme=len(extreme),
        )

    def _reliability_buckets(
        self, results: list[ModelResult]
    ) -> list[tuple[float, float, int]]:
        """Returns list of (mean_confidence_in_bucket, fraction_correct, n_samples)."""
        buckets: list[list[ModelResult]] = [[] for _ in range(self.n_buckets)]
        for r in results:
            idx = min(int(r.predicted_score * self.n_buckets), self.n_buckets - 1)
            buckets[idx].append(r)

        out = []
        for bucket in buckets:
            if not bucket:
                continue
            mean_conf = sum(r.predicted_score for r in bucket) / len(bucket)
            frac_corr = sum(1 for r in bucket if r.actual_correct) / len(bucket)
            out.append((round(mean_conf, 3), round(frac_corr, 3), len(bucket)))
        return out

    def _ece(self, results: list[ModelResult]) -> float:
        if not results:
            return 0.0
        buckets: list[list[ModelResult]] = [[] for _ in range(self.n_buckets)]
        for r in results:
            idx = min(int(r.predicted_score * self.n_buckets), self.n_buckets - 1)
            buckets[idx].append(r)
        N = len(results)
        ece = 0.0
        for bucket in buckets:
            if not bucket:
                continue
            mean_conf = sum(r.predicted_score for r in bucket) / len(bucket)
            frac_corr = sum(1 for r in bucket if r.actual_correct) / len(bucket)
            ece += (len(bucket) / N) * abs(mean_conf - frac_corr)
        return ece

    def _accuracy(self, results: list[ModelResult]) -> float:
        if not results:
            return 0.0
        return sum(1 for r in results if r.actual_correct) / len(results)

    def _overconfidence(self, results: list[ModelResult]) -> float:
        if not results:
            return 0.0
        mean_conf = sum(r.predicted_score for r in results) / len(results)
        acc = self._accuracy(results)
        return mean_conf - acc


# ---------------------------------------------------------------------------
# Model comparison runner
# ---------------------------------------------------------------------------

class ModelComparison:
    """
    Runs multiple prediction strategies against a query set and compares their
    calibration on normal vs. extreme queries.

    Paper analogy: the authors compare Aurora, GenCast, Pangu, GraphCast on
    normal precipitation and extreme precipitation events. Here we compare
    retrieval strategies on normal vs. extreme queries.
    """

    def __init__(
        self,
        corpus: list[str],
        extreme_percentile: float = 0.85,
        n_buckets: int = 10,
    ):
        self._classifier = ExtremeQueryClassifier(corpus, extreme_percentile)
        self._calibrator = ReliabilityCalibrator(n_buckets)

    def run(
        self,
        queries: list[str],
        model_names: list[str],
    ) -> dict[str, CalibrationReport]:
        """
        Returns a CalibrationReport per selected model.

        queries: list of query strings to evaluate
        model_names: names of models from BUILTIN_MODELS to compare
        """
        classified = self._classifier.classify(queries)
        reports: dict[str, CalibrationReport] = {}

        for model_name in model_names:
            model = BUILTIN_MODELS.get(model_name)
            if model is None:
                continue

            results: list[ModelResult] = []
            for query, extremity, q_type in classified:
                confidence, is_correct = model.predict(query, extremity)
                results.append(ModelResult(
                    model_name=model_name,
                    query=query,
                    query_type=q_type,
                    predicted_score=confidence,
                    actual_correct=is_correct,
                    percentile=extremity,
                ))

            reports[model_name] = self._calibrator.compute(results)

        return reports

    def classify_queries(self, queries: list[str]) -> list[tuple[str, float, str]]:
        """Expose query classification for the UI."""
        return self._classifier.classify(queries)
