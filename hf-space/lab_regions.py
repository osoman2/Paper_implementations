"""
Lab regions (the "continents" of the Lab). Plain Python, no gradio import, so tools outside the
Space (tools/kb.py, the daily agent) can classify POCs the same way the Space does.
"""

import re

REGIONS = [
    ("agents", "Agents & Orchestration"),
    ("retrieval", "Retrieval & RAG"),
    ("efficiency", "Inference Efficiency"),
    ("security", "Agent Security"),
    ("evaluation", "Evaluation & Interpretability"),
    ("signals", "Signals & Optimisation"),
]
REGION_LABEL = dict(REGIONS)

# Curated regions, kept in sync with the Lab on vectorridgelabs.com.
REGION_BY_CODE = {
    "POC-001": "retrieval", "POC-002": "evaluation", "POC-003": "signals", "POC-004": "retrieval",
    "POC-005": "evaluation", "POC-006": "signals", "POC-007": "retrieval", "POC-008": "signals",
    "POC-009": "agents", "POC-010": "evaluation", "POC-011": "evaluation", "POC-012": "retrieval",
    "POC-013": "retrieval", "POC-014": "agents", "POC-015": "signals", "POC-016": "security",
    "POC-017": "security", "POC-018": "security", "POC-019": "security", "POC-020": "efficiency",
    "POC-021": "retrieval", "POC-022": "efficiency", "POC-023": "agents", "POC-024": "efficiency",
    "POC-025": "agents", "POC-026": "security", "POC-027": "agents", "POC-028": "agents",
    "POC-029": "evaluation", "POC-030": "evaluation", "POC-031": "agents", "POC-032": "security",
    "POC-033": "agents", "POC-034": "retrieval", "POC-035": "efficiency", "POC-036": "efficiency",
    "POC-037": "agents", "POC-038": "agents", "POC-039": "agents", "POC-040": "agents",
    "POC-041": "evaluation", "POC-042": "agents", "POC-043": "agents", "POC-044": "agents",
    "POC-045": "retrieval", "POC-046": "agents", "POC-047": "agents", "POC-048": "retrieval",
    "POC-049": "efficiency", "POC-050": "agents", "POC-051": "efficiency",
}

_KEYWORDS = [
    ("security", r"secur|shield|escape|jailbreak|injection|refusal|attack|sandbox|privacy"),
    ("retrieval", r"\brag\b|retriev|search|embedding|index|rerank|knowledge graph"),
    ("efficiency", r"\bkv\b|cache|cheaper|compute|latency|quantiz|\bmoe\b|cost"),
    ("agents", r"agent|harness|swarm|orchestrat|memory|context|self-improv"),
    ("evaluation", r"calibrat|attribution|evaluat|benchmark|interpretab|verif"),
]


def region_for(code: str, label: str) -> str:
    if code in REGION_BY_CODE:
        return REGION_BY_CODE[code]
    low = label.lower()
    for region, pattern in _KEYWORDS:
        if re.search(pattern, low):
            return region
    return "signals"
