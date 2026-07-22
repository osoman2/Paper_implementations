"""
PromptShield POC — Privacy-preserving LLM proxy.

Based on: "PromptShield: Protecting User Privacy in AI Chatbots"
(Asimine et al., Technical University of Denmark, Software, 2026).

Pipeline:
  user prompt → regex + heuristic NER → entity categorization
              → user-controlled sanitization → template reformulation
              → HITL review → LLM (only approved text crosses boundary)

Key insight: The system is not a firewall — it is a collaborative editor.
Rather than blocking queries, it restructures them so the LLM still receives
enough context to answer usefully while PII stays on the user's machine.
"""

from .detector import detect, DetectionResult, DetectedEntity, PrivacyCategory
from .sanitizer import reformulate, privacy_report

__all__ = [
    "detect",
    "DetectionResult",
    "DetectedEntity",
    "PrivacyCategory",
    "reformulate",
    "privacy_report",
]
