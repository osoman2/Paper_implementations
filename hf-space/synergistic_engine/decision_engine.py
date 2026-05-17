from __future__ import annotations
from dataclasses import dataclass
from .event_stream import Event, EventStream, EventType
from .rule_engine import Rule, RuleEngine


ACTION_LABELS = {
    "USE_SPARSE_RETRIEVAL":  ("BM25 / Sparse", "#2196F3"),
    "USE_DENSE_RETRIEVAL":   ("Dense / Semantic", "#9C27B0"),
    "USE_HYBRID_SEARCH":     ("Hybrid (α≈0.5)", "#4CAF50"),
    "FALLBACK_RETRIEVAL":    ("Fallback", "#F44336"),
    "RERANK_RESULTS":        ("Cross-Encoder Rerank", "#FF9800"),
    "REGENERATE_HIGH_TEMP":  ("Regen (high-temp)", "#E91E63"),
    "SERVE_FROM_CACHE":      ("Cache Hit", "#607D8B"),
}


@dataclass
class DecisionResult:
    action: str | None
    fired_rules: list[str]
    all_evaluations: list[tuple[str, bool, str, int]]
    explanation: str


class DecisionEngine:
    def __init__(self):
        self.stream = EventStream(window_size=20)
        self.rule_engine = RuleEngine()

    def process_event(self, event: Event) -> DecisionResult:
        self.stream.emit(event)
        evaluations = self.rule_engine.evaluate(self.stream)
        fired = [(r.name, r.action) for r, ok in evaluations if ok]
        all_evals = [
            (r.name, ok, r.action, r.priority)
            for r, ok in evaluations
        ]
        action = evaluations[0][0].action if evaluations and evaluations[0][1] else None
        for r, ok in evaluations:
            if ok:
                action = r.action
                break
        label, _ = ACTION_LABELS.get(action, (action, "#999")) if action else ("No action", "#999")
        explanation = (
            f"**Decision: {label}**\n\n"
            f"Fired rules: {', '.join(r for r, _ in fired) if fired else '_none_'}\n\n"
            f"Event stream depth: {len(self.stream.recent())} events"
        )
        return DecisionResult(
            action=action,
            fired_rules=[r for r, _ in fired],
            all_evaluations=all_evals,
            explanation=explanation,
        )

    def reset(self) -> None:
        self.stream.clear()
        for rule in self.rule_engine.rules:
            rule.fired_count = 0
