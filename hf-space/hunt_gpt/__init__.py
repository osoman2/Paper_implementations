"""
HuntGPT POC — RAG anomaly detection with XAI + conversational assistant.

Based on: "HuntGPT: Integrating Machine Learning-Based Anomaly Detection and
Explainable AI with Large Language Models" (Ali et al., University of Oulu,
Telecom, 2026).

Pipeline (paper → RAG adaptation):
  Network traffic     → RAG query + retrieval metrics
  Random Forest       → RF hallucination-risk classifier
  SHAP/LIME           → feature attribution for flagged queries
  LLM assistant       → rule-based conversational explanation (no API key needed)
"""

from .detector import (
    RAGQueryRecord,
    DetectionResult,
    classify_query,
    FEATURE_NAMES,
    FEATURE_DESCRIPTIONS,
    get_detector,
)
from .explainer import explain, Explanation
from .assistant import generate_analysis, answer_question

__all__ = [
    "RAGQueryRecord",
    "DetectionResult",
    "classify_query",
    "explain",
    "Explanation",
    "generate_analysis",
    "answer_question",
    "FEATURE_NAMES",
    "FEATURE_DESCRIPTIONS",
    "get_detector",
]
