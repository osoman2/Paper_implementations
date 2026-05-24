"""
Extreme Event AI — POC module.

Implements model reliability calibration for extreme/edge-case inputs,
inspired by the paper:
"Predictability of precipitation extremes in the Western United States
 and the U.S. Gulf Coast using AI weather models"

The paper evaluates multiple AI models (Aurora, GenCast, Pangu, GraphCast)
on extreme precipitation events, showing that AI models can be overconfident
or underconfident on tail-distribution inputs vs. normal ones.

RAG/LLM engineering analogy:
  - "AI weather models" → retrieval/prediction strategies (BM25, dense, hybrid, reranker)
  - "extreme precipitation events" → extreme/edge-case queries (very rare terms,
    adversarial phrasing, out-of-distribution inputs)
  - "reliability calibration" → does confidence ∝ actual accuracy?

Core classes:
  ExtremeQueryClassifier — splits queries into "normal" vs "extreme" buckets
  ReliabilityCalibrator  — computes ECE, reliability diagram data, overconfidence
  ModelComparison        — compares multiple prediction strategies
"""

from .calibrator import (
    ExtremeQueryClassifier,
    ReliabilityCalibrator,
    ModelComparison,
    ModelResult,
    CalibrationReport,
    BUILTIN_MODELS,
)

__all__ = [
    "ExtremeQueryClassifier",
    "ReliabilityCalibrator",
    "ModelComparison",
    "ModelResult",
    "CalibrationReport",
    "BUILTIN_MODELS",
]
