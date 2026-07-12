"""
HuntGPT POC — SHAP/LIME-style feature attribution.

Produces local and global explanations for the RAG anomaly classifier
without requiring the shap or lime packages — uses perturbation-based
approximations that replicate the spirit of both methods:

  LIME approximation  — perturb feature values around the query point,
                        observe prediction changes, fit a linear local model.
  SHAP approximation  — estimate marginal contribution of each feature by
                        computing predictions with/without each feature (mean
                        substitution for absent features), averaged over a
                        small reference sample.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .detector import (
    FEATURE_NAMES,
    FEATURE_DESCRIPTIONS,
    RAGAnomalyDetector,
    _synthetic_dataset,
    get_detector,
)


@dataclass
class FeatureContribution:
    name: str
    value: float
    contribution: float     # positive = pushes toward anomaly
    description: str


@dataclass
class Explanation:
    """Local explanation for one detection result."""
    lime_contributions: list[FeatureContribution]
    shap_contributions: list[FeatureContribution]
    global_importances: dict[str, float]

    def top_lime(self, n: int = 5) -> list[FeatureContribution]:
        return sorted(self.lime_contributions, key=lambda c: abs(c.contribution), reverse=True)[:n]

    def top_shap(self, n: int = 5) -> list[FeatureContribution]:
        return sorted(self.shap_contributions, key=lambda c: abs(c.contribution), reverse=True)[:n]


# ---------------------------------------------------------------------------
# Reference dataset (used by SHAP approximation)
# ---------------------------------------------------------------------------

def _reference_data(n: int = 50) -> np.ndarray:
    X, _ = _synthetic_dataset(n=n * 4, seed=7)
    return X[:n]


# ---------------------------------------------------------------------------
# LIME approximation
# ---------------------------------------------------------------------------

def lime_explain(
    x: np.ndarray,
    detector: RAGAnomalyDetector,
    n_samples: int = 200,
    noise_scale: float = 0.25,
    seed: int = 0,
) -> list[FeatureContribution]:
    """
    Perturb x, collect predictions, fit a weighted linear model to estimate
    how each feature pushes the anomaly score locally.
    """
    rng = np.random.default_rng(seed)
    n_feat = len(x)

    # Scaled reference to estimate feature std
    x_scaled = detector.scaler.transform(x.reshape(1, -1))[0]

    # Generate perturbed samples in scaled space
    noise = rng.normal(0, noise_scale, size=(n_samples, n_feat))
    perturbed_scaled = x_scaled + noise

    # Predict on perturbed samples
    probs = detector.clf.predict_proba(perturbed_scaled)[:, 1]

    # Weight by proximity (cosine kernel)
    distances = np.linalg.norm(noise, axis=1)
    weights = np.exp(-0.5 * (distances / distances.std()) ** 2)

    # Weighted linear regression: contribution[i] ≈ w_i * noise[:, i]
    contribs = []
    for i, name in enumerate(FEATURE_NAMES):
        cov = np.cov(noise[:, i], probs, aweights=weights + 1e-8)
        var = np.var(noise[:, i]) + 1e-8
        coeff = float(cov[0, 1] / var)
        # Scale to original feature units
        contribs.append(FeatureContribution(
            name=name,
            value=float(x[i]),
            contribution=coeff,
            description=FEATURE_DESCRIPTIONS[name],
        ))

    return contribs


# ---------------------------------------------------------------------------
# SHAP approximation (marginal contribution via mean substitution)
# ---------------------------------------------------------------------------

def shap_explain(
    x: np.ndarray,
    detector: RAGAnomalyDetector,
    n_ref: int = 40,
) -> list[FeatureContribution]:
    """
    For each feature, estimate its SHAP value as the mean difference between
    predictions with the feature present vs. replaced by its reference mean.
    """
    ref = _reference_data(n_ref)
    ref_means = ref.mean(axis=0)

    # Baseline: prediction with all features replaced by reference means
    baseline_x = ref_means.copy()
    baseline_scaled = detector.scaler.transform(baseline_x.reshape(1, -1))
    baseline_prob = float(detector.clf.predict_proba(baseline_scaled)[0][1])

    # Full prediction
    full_scaled = detector.scaler.transform(x.reshape(1, -1))
    full_prob = float(detector.clf.predict_proba(full_scaled)[0][1])

    contribs = []
    for i, name in enumerate(FEATURE_NAMES):
        # Coalition WITHOUT feature i: replace feature i with reference mean
        x_without = x.copy()
        x_without[i] = ref_means[i]
        xw_scaled = detector.scaler.transform(x_without.reshape(1, -1))
        prob_without = float(detector.clf.predict_proba(xw_scaled)[0][1])

        shap_val = full_prob - prob_without
        contribs.append(FeatureContribution(
            name=name,
            value=float(x[i]),
            contribution=shap_val,
            description=FEATURE_DESCRIPTIONS[name],
        ))

    return contribs


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def explain(x: np.ndarray, detector: Optional[RAGAnomalyDetector] = None) -> Explanation:
    if detector is None:
        detector = get_detector()

    lime_contribs = lime_explain(x, detector)
    shap_contribs = shap_explain(x, detector)
    global_imp    = detector.feature_importances()

    return Explanation(
        lime_contributions=lime_contribs,
        shap_contributions=shap_contribs,
        global_importances=global_imp,
    )
