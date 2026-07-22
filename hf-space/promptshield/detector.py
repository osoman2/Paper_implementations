"""
PromptShield — NER & regex entity detection pipeline.

Adapted from: "PromptShield: Protecting User Privacy in AI Chatbots"
(Asimine et al., Technical University of Denmark, Software, 2026).

Pipeline:
  prompt → normalization → regex layer → NER-style heuristic layer
          → entity span consolidation → privacy category mapping
          → dominant topic classification
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


class PrivacyCategory(str, Enum):
    NAME       = "Personal Name"
    EMAIL      = "Email Address"
    PHONE      = "Phone Number"
    ADDRESS    = "Physical Address"
    CREDENTIAL = "Credential / Account"
    MEDICAL    = "Medical Information"
    FINANCIAL  = "Financial Information"
    LOCATION   = "Location"
    DATE_DOB   = "Date / Date of Birth"
    IDENTIFIER = "Identifier / ID Number"


@dataclass
class DetectedEntity:
    text:      str
    category:  PrivacyCategory
    start:     int
    end:       int
    keep:      bool = False  # user decision — default: remove

    def placeholder(self) -> str:
        return f"[{self.category.name}]"


@dataclass
class DetectionResult:
    original:    str
    entities:    List[DetectedEntity] = field(default_factory=list)
    topic:       str = "general"

    def sanitized(self) -> str:
        """Return prompt with all entities marked keep=False replaced."""
        text = self.original
        for ent in sorted(self.entities, key=lambda e: e.start, reverse=True):
            if not ent.keep:
                text = text[: ent.start] + ent.placeholder() + text[ent.end :]
        return text

    def category_summary(self) -> dict:
        summary: dict = {}
        for ent in self.entities:
            summary.setdefault(ent.category.value, []).append(ent.text)
        return summary


# ---------------------------------------------------------------------------
# Regex patterns (layer 1 — deterministic)
# ---------------------------------------------------------------------------
_PATTERNS: List[Tuple[re.Pattern, PrivacyCategory]] = [
    # Email
    (re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"),
     PrivacyCategory.EMAIL),
    # Phone (US-style + international)
    (re.compile(r"(?:\+?\d{1,3}[\s\-\.])?(?:\(?\d{2,4}\)?[\s\-\.])\d{3,4}[\s\-\.]\d{4}"),
     PrivacyCategory.PHONE),
    # Credit / debit card
    (re.compile(r"\b(?:\d[ \-]?){13,16}\b"),
     PrivacyCategory.FINANCIAL),
    # SSN
    (re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"),
     PrivacyCategory.IDENTIFIER),
    # Date of birth keywords
    (re.compile(
        r"\b(?:born\s+on|date\s+of\s+birth|dob)[:\s]+\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b",
        re.IGNORECASE,
    ), PrivacyCategory.DATE_DOB),
    # ISO/slash dates
    (re.compile(r"\b\d{4}[/\-]\d{2}[/\-]\d{2}\b"), PrivacyCategory.DATE_DOB),
    # Street address heuristic
    (re.compile(
        r"\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+"
        r"(?:St(?:reet)?|Ave(?:nue)?|Blvd|Rd|Dr|Ln|Ct|Pl|Way|Pkwy)\.?\b",
        re.IGNORECASE,
    ), PrivacyCategory.ADDRESS),
    # Medical keywords
    (re.compile(
        r"\b(?:diagnosed with|suffering from|taking|prescribed|allergic to|"
        r"my (?:doctor|therapist|psychiatrist)|mental health|anxiety|depression|"
        r"diabetes|hypertension|HIV|cancer|medication|dosage)\b",
        re.IGNORECASE,
    ), PrivacyCategory.MEDICAL),
    # Financial keywords
    (re.compile(
        r"\b(?:bank account|routing number|IBAN|account number|salary|income|"
        r"tax return|social security|retirement account|401k|IRA)\b",
        re.IGNORECASE,
    ), PrivacyCategory.FINANCIAL),
    # Credentials
    (re.compile(
        r"\b(?:password|passwd|api[\s_\-]?key|secret[\s_\-]?key|access[\s_\-]?token|"
        r"auth[\s_\-]?token|login|credentials)\b",
        re.IGNORECASE,
    ), PrivacyCategory.CREDENTIAL),
]

# ---------------------------------------------------------------------------
# Heuristic NER layer (layer 2 — statistical-style rules without XLM-RoBERTa)
# ---------------------------------------------------------------------------
# Common name prefixes that indicate a personal name follows
_NAME_PREFIXES = re.compile(
    r"(?:my name is|I am|I'm|call me|this is|patient:?|"
    r"Mr\.?|Ms\.?|Mrs\.?|Dr\.?|Prof\.?)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
)
_LOCATION_PREP = re.compile(
    r"\b(?:in|at|near|from|to|visit(?:ing)?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,1})\b"
)


def _heuristic_names(text: str) -> List[Tuple[int, int, PrivacyCategory]]:
    spans = []
    for m in _NAME_PREFIXES.finditer(text):
        g = m.group(1)
        s = m.start(1)
        spans.append((s, s + len(g), PrivacyCategory.NAME))
    return spans


def _heuristic_locations(text: str) -> List[Tuple[int, int, PrivacyCategory]]:
    spans = []
    skip = {"the", "a", "an", "my", "our", "their", "this", "that"}
    for m in _LOCATION_PREP.finditer(text):
        g = m.group(1)
        if g.lower() not in skip and len(g) > 2:
            s = m.start(1)
            spans.append((s, s + len(g), PrivacyCategory.LOCATION))
    return spans


# ---------------------------------------------------------------------------
# Topic classifier (keyword-based, maps to reformulation template key)
# ---------------------------------------------------------------------------
_TOPIC_KEYWORDS: dict = {
    "medical":    ["doctor", "symptom", "diagnosis", "prescri", "treatment", "hospital",
                   "medication", "allerg", "pain", "fever", "anxiety", "depression"],
    "legal":      ["lawyer", "attorney", "legal", "lawsuit", "court", "contract",
                   "rights", "sue", "appeal", "charges"],
    "financial":  ["bank", "invest", "tax", "salary", "loan", "mortgage", "credit",
                   "budget", "financial", "insurance"],
    "technical":  ["code", "debug", "error", "stack", "api", "server", "deploy",
                   "repository", "function", "class"],
    "personal":   ["relationship", "family", "friend", "personal", "advice",
                   "feel", "emotion", "private"],
}


def classify_topic(text: str) -> str:
    text_lower = text.lower()
    scores = {t: sum(1 for kw in kws if kw in text_lower)
              for t, kws in _TOPIC_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(prompt: str) -> DetectionResult:
    """Run full detection pipeline on a prompt string."""
    spans: List[Tuple[int, int, PrivacyCategory]] = []

    # Layer 1: regex
    for pat, cat in _PATTERNS:
        for m in pat.finditer(prompt):
            spans.append((m.start(), m.end(), cat))

    # Layer 2: heuristics
    spans.extend(_heuristic_names(prompt))
    spans.extend(_heuristic_locations(prompt))

    # Deduplicate & sort
    seen: set = set()
    deduped = []
    for s, e, cat in sorted(spans, key=lambda x: (x[0], -(x[1] - x[0]))):
        if any(s < es and e > es for es, ee, _ in deduped):
            continue
        key = (s, e)
        if key not in seen:
            seen.add(key)
            deduped.append((s, e, cat))

    entities = [
        DetectedEntity(text=prompt[s:e], category=cat, start=s, end=e)
        for s, e, cat in sorted(deduped, key=lambda x: x[0])
    ]

    return DetectionResult(
        original=prompt,
        entities=entities,
        topic=classify_topic(prompt),
    )
