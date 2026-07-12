"""
HuntGPT POC — RAG Anomaly Detector.

Maps the HuntGPT pipeline to AI engineering:
  Paper layer          → RAG layer
  ─────────────────────────────────────────────────────
  Network traffic      → RAG query + retrieval metrics
  Random Forest        → RF classifier for hallucination risk
  SHAP/LIME            → feature attribution for flagged queries
  LLM assistant        → conversational explanation of the alert

The classifier is trained on synthetic RAG quality metrics.
Features describe the query and its retrieval context; the label
is 0 = normal (low hallucination risk), 1 = anomalous (high risk).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Feature schema — 10 numeric features per RAG query/retrieval pair
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "query_length",           # number of tokens in the query
    "query_specificity",      # ratio of rare/domain terms (0–1)
    "entity_count",           # named-entity count detected in query
    "top1_similarity",        # cosine similarity of best retrieved doc
    "score_gap",              # top1_sim − top2_sim (confidence margin)
    "bm25_dense_ratio",       # BM25 rank / dense rank for top doc
    "retrieved_doc_length",   # token length of top retrieved document
    "query_doc_overlap",      # unigram overlap between query and top doc
    "answer_span_length",     # LLM-generated answer token length
    "retrieval_latency_ms",   # time to retrieve (proxy for index load)
]

FEATURE_DESCRIPTIONS = {
    "query_length":        "Query token count — very short/long queries are higher risk",
    "query_specificity":   "Fraction of domain-specific / rare terms in query",
    "entity_count":        "Number of named entities — entity-heavy queries need exact matches",
    "top1_similarity":     "Top-1 cosine similarity — low similarity → hallucination risk",
    "score_gap":           "Gap between 1st and 2nd retrieval scores — small gap = ambiguous",
    "bm25_dense_ratio":    "Agreement between sparse/dense rankers — divergence signals noise",
    "retrieved_doc_length":"Length of top retrieved doc — very short docs may lack context",
    "query_doc_overlap":   "Lexical overlap — low overlap means retrieved doc may be off-topic",
    "answer_span_length":  "Answer token length — very long answers may hallucinate",
    "retrieval_latency_ms":"Retrieval latency — high latency may correlate with index issues",
}

# ---------------------------------------------------------------------------
# Synthetic training data
# ---------------------------------------------------------------------------

def _synthetic_dataset(n: int = 800, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic RAG quality dataset (n samples, 10 features, binary label)."""
    rng = np.random.default_rng(seed)

    # Normal samples: high similarity, good overlap, moderate query length
    n_normal = n // 2
    normal = np.column_stack([
        rng.integers(8, 30, n_normal),               # query_length
        rng.uniform(0.1, 0.5, n_normal),             # query_specificity
        rng.integers(0, 4, n_normal),                # entity_count
        rng.uniform(0.55, 0.95, n_normal),           # top1_similarity (high)
        rng.uniform(0.05, 0.25, n_normal),           # score_gap (decent margin)
        rng.uniform(0.8, 1.3, n_normal),             # bm25_dense_ratio (agree)
        rng.integers(80, 400, n_normal),             # retrieved_doc_length
        rng.uniform(0.25, 0.70, n_normal),           # query_doc_overlap (good)
        rng.integers(40, 200, n_normal),             # answer_span_length
        rng.uniform(20, 120, n_normal),              # retrieval_latency_ms
    ])

    # Anomalous samples: low similarity, poor overlap, extreme query/answer lengths
    n_anom = n - n_normal
    anomalous = np.column_stack([
        rng.choice(
            np.concatenate([rng.integers(1, 5, n_anom // 2),
                            rng.integers(60, 120, n_anom - n_anom // 2)]),
            n_anom, replace=False,
        ),                                           # query_length (too short or too long)
        rng.uniform(0.6, 1.0, n_anom),              # query_specificity (very domain-heavy)
        rng.integers(3, 10, n_anom),                # entity_count (many entities)
        rng.uniform(0.05, 0.45, n_anom),            # top1_similarity (low)
        rng.uniform(0.00, 0.04, n_anom),            # score_gap (near-zero → ambiguous)
        rng.uniform(2.0, 5.0, n_anom),              # bm25_dense_ratio (diverge)
        rng.integers(5, 40, n_anom),                # retrieved_doc_length (too short)
        rng.uniform(0.00, 0.15, n_anom),            # query_doc_overlap (near zero)
        rng.integers(300, 600, n_anom),             # answer_span_length (very long)
        rng.uniform(200, 800, n_anom),              # retrieval_latency_ms (high)
    ])

    X = np.vstack([normal, anomalous]).astype(float)
    y = np.array([0] * n_normal + [1] * n_anom)

    # Shuffle
    idx = rng.permutation(n)
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Trained classifier (singleton, fitted once at import)
# ---------------------------------------------------------------------------

class RAGAnomalyDetector:
    """Random Forest classifier for RAG hallucination / retrieval anomaly detection."""

    def __init__(self, n_estimators: int = 100, seed: int = 42):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required")
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            random_state=seed,
            class_weight="balanced",
        )
        self.scaler = StandardScaler()
        self._fit()

    def _fit(self):
        X, y = _synthetic_dataset()
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)

    def predict(self, features: np.ndarray) -> tuple[int, float]:
        """Return (label, anomaly_probability) for a single feature vector."""
        x = self.scaler.transform(features.reshape(1, -1))
        label = int(self.clf.predict(x)[0])
        prob  = float(self.clf.predict_proba(x)[0][1])
        return label, prob

    def feature_importances(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.clf.feature_importances_))


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

@dataclass
class RAGQueryRecord:
    """A single RAG query with its retrieval metrics."""
    query_length: float = 12.0
    query_specificity: float = 0.3
    entity_count: float = 1.0
    top1_similarity: float = 0.75
    score_gap: float = 0.15
    bm25_dense_ratio: float = 1.0
    retrieved_doc_length: float = 200.0
    query_doc_overlap: float = 0.40
    answer_span_length: float = 100.0
    retrieval_latency_ms: float = 60.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.query_length,
            self.query_specificity,
            self.entity_count,
            self.top1_similarity,
            self.score_gap,
            self.bm25_dense_ratio,
            self.retrieved_doc_length,
            self.query_doc_overlap,
            self.answer_span_length,
            self.retrieval_latency_ms,
        ])


@dataclass
class DetectionResult:
    label: int               # 0 = normal, 1 = anomalous
    anomaly_prob: float
    record: RAGQueryRecord
    feature_values: dict[str, float] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        if self.anomaly_prob >= 0.7:
            return "🔴 HIGH RISK"
        if self.anomaly_prob >= 0.4:
            return "🟡 ELEVATED RISK"
        return "🟢 NORMAL"

    @property
    def risk_pct(self) -> float:
        return round(self.anomaly_prob * 100, 1)


_DETECTOR: Optional[RAGAnomalyDetector] = None


def get_detector() -> RAGAnomalyDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = RAGAnomalyDetector()
    return _DETECTOR


def classify_query(record: RAGQueryRecord) -> DetectionResult:
    det = get_detector()
    x = record.to_array()
    label, prob = det.predict(x)
    fv = dict(zip(FEATURE_NAMES, x.tolist()))
    return DetectionResult(label=label, anomaly_prob=prob, record=record, feature_values=fv)
