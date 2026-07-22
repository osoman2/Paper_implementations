"""
PromptShield — sanitization & template-based reformulation.

Adapted from: "PromptShield: Protecting User Privacy in AI Chatbots"
(Asimine et al., Technical University of Denmark, Software, 2026).

After entity detection, this module:
1. Produces the sanitized prompt (placeholders in place of PII)
2. Detects intent / task type from the sanitized text
3. Selects a reformulation template for the dominant task
4. Fills the template with non-PII context to preserve LLM utility
"""

from __future__ import annotations
import re
from typing import Optional
from .detector import DetectionResult, PrivacyCategory

# ---------------------------------------------------------------------------
# Intent / task classifier → selects reformulation template
# ---------------------------------------------------------------------------
_INTENT_PATTERNS: dict[str, list] = {
    "question":      [r"\b(?:what|how|why|when|where|who|which|can you|could you|"
                      r"please explain|tell me about)\b"],
    "diagnosis":     [r"\b(?:diagnosed|diagnosis|symptom|do i have|am i sick|"
                      r"is it possible|could this be)\b"],
    "legal_advice":  [r"\b(?:legal|lawsuit|sue|rights|attorney|contract|court)\b"],
    "financial_help":[r"\b(?:invest|tax|loan|bank|credit|financial|salary|budget)\b"],
    "code_help":     [r"\b(?:code|function|debug|error|api|implement|write a)\b"],
    "personal_advice":[r"\b(?:relationship|advice|feel|personal|should i|help me with)\b"],
}

# Template library — each preserves query intent without PII
_TEMPLATES: dict[str, str] = {
    "question": (
        "I have a question about {topic}. "
        "Specifically, I would like to understand: {core_query}. "
        "Please provide a clear and concise explanation."
    ),
    "diagnosis": (
        "A person is experiencing certain symptoms related to {topic}. "
        "Could you provide general information about possible causes and "
        "when to seek professional medical advice? "
        "Note: This is for informational purposes only."
    ),
    "legal_advice": (
        "I have a question about {topic} from a legal perspective. "
        "{core_query} "
        "Please provide general legal information, not specific legal advice."
    ),
    "financial_help": (
        "I need guidance on a financial matter related to {topic}. "
        "{core_query} "
        "Please provide general financial information and considerations."
    ),
    "code_help": (
        "I need help with a programming task involving {topic}. "
        "{core_query} "
        "Please provide a code example or explanation."
    ),
    "personal_advice": (
        "I am seeking general advice about {topic}. "
        "{core_query} "
        "Please provide thoughtful and balanced guidance."
    ),
    "general": (
        "I have a request related to {topic}. "
        "{core_query} "
        "Please assist with the above while keeping your response clear and concise."
    ),
}


def _classify_intent(text: str) -> str:
    text_lower = text.lower()
    for intent, patterns in _INTENT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                return intent
    return "general"


def _extract_core_query(sanitized: str) -> str:
    """Extract a non-PII sentence fragment for use in template."""
    # Keep placeholder tags as readable labels rather than stripping them
    readable = re.sub(r"\[([A-Z_]+)\]", r"<\1>", sanitized).strip()
    # Collapse multiple spaces left by inline replacements
    readable = re.sub(r"\s{2,}", " ", readable)
    return readable[:150] + ("..." if len(readable) > 150 else "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reformulate(result: DetectionResult) -> str:
    """Generate a template-based reformulation of the sanitized prompt."""
    sanitized = result.sanitized()
    intent = _classify_intent(result.original)
    template = _TEMPLATES.get(intent, _TEMPLATES["general"])
    core_query = _extract_core_query(sanitized)
    topic = result.topic.replace("_", " ")
    return template.format(topic=topic, core_query=core_query)


def privacy_report(result: DetectionResult) -> str:
    """Build a Markdown privacy report for display in Gradio."""
    if not result.entities:
        return "✅ **No sensitive information detected.** Your prompt appears safe to send as-is."

    lines = [
        f"### 🔍 Privacy Scan — {len(result.entities)} entity/entities detected\n",
        f"**Detected topic:** `{result.topic}`\n",
        "| # | Detected Text | Privacy Category | Action |",
        "|---|---------------|------------------|--------|",
    ]
    for i, ent in enumerate(result.entities, 1):
        action = "✅ Keep" if ent.keep else "🚫 Remove"
        lines.append(f"| {i} | `{ent.text}` | {ent.category.value} | {action} |")

    removed = [e for e in result.entities if not e.keep]
    kept    = [e for e in result.entities if e.keep]

    lines.append("")
    lines.append(f"**Entities removed:** {len(removed)}  |  "
                 f"**Entities kept:** {len(kept)}")

    category_counts: dict = {}
    for e in removed:
        category_counts[e.category.value] = category_counts.get(e.category.value, 0) + 1
    if category_counts:
        lines.append("\n**Privacy categories redacted:**")
        for cat, cnt in sorted(category_counts.items()):
            lines.append(f"- {cat}: {cnt}")

    return "\n".join(lines)
