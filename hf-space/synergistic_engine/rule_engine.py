from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
from .event_stream import Event, EventStream, EventType


@dataclass
class Rule:
    name: str
    description: str
    condition: Callable[[EventStream], bool]
    action: str
    priority: int = 5
    fired_count: int = 0

    def evaluate(self, stream: EventStream) -> bool:
        try:
            return self.condition(stream)
        except Exception:
            return False


class RuleEngine:
    def __init__(self):
        self._rules: list[Rule] = []
        self._load_defaults()

    def _load_defaults(self) -> None:
        self._rules = [
            Rule(
                name="sparse_preferred",
                description="Query contains rare/technical terms → prefer BM25 sparse retrieval",
                condition=lambda s: (
                    s.last_of_type(EventType.QUERY) is not None
                    and s.last_of_type(EventType.QUERY).payload.get("idf_score", 0) > 0.65
                ),
                action="USE_SPARSE_RETRIEVAL",
                priority=8,
            ),
            Rule(
                name="dense_preferred",
                description="Conceptual/semantic query → prefer dense vector retrieval",
                condition=lambda s: (
                    s.last_of_type(EventType.QUERY) is not None
                    and s.last_of_type(EventType.QUERY).payload.get("idf_score", 0.5) < 0.35
                ),
                action="USE_DENSE_RETRIEVAL",
                priority=8,
            ),
            Rule(
                name="escalate_on_errors",
                description="≥ 2 errors in last 5 events → escalate to fallback retrieval",
                condition=lambda s: s.count_type(EventType.ERROR, last_n=5) >= 2,
                action="FALLBACK_RETRIEVAL",
                priority=10,
            ),
            Rule(
                name="rerank_after_tool",
                description="Tool call followed by context retrieval → trigger cross-encoder reranking",
                condition=lambda s: (
                    s.last_of_type(EventType.TOOL_CALLED) is not None
                    and s.last_of_type(EventType.CONTEXT_RETRIEVED) is not None
                    and s.last_of_type(EventType.CONTEXT_RETRIEVED).timestamp
                    > s.last_of_type(EventType.TOOL_CALLED).timestamp
                ),
                action="RERANK_RESULTS",
                priority=6,
            ),
            Rule(
                name="low_confidence_regen",
                description="Response generated with low confidence → regenerate with higher temperature",
                condition=lambda s: (
                    s.last_of_type(EventType.RESPONSE_GENERATED) is not None
                    and s.last_of_type(EventType.RESPONSE_GENERATED).payload.get("confidence", 1.0) < 0.5
                ),
                action="REGENERATE_HIGH_TEMP",
                priority=7,
            ),
            Rule(
                name="hybrid_balanced",
                description="Mid-range IDF query → use balanced hybrid search (alpha ≈ 0.5)",
                condition=lambda s: (
                    s.last_of_type(EventType.QUERY) is not None
                    and 0.35 <= s.last_of_type(EventType.QUERY).payload.get("idf_score", 0.5) <= 0.65
                ),
                action="USE_HYBRID_SEARCH",
                priority=5,
            ),
            Rule(
                name="cache_hit_skip",
                description="Same query type seen ≥ 3 times → return cached result",
                condition=lambda s: s.count_type(EventType.QUERY, last_n=10) >= 3,
                action="SERVE_FROM_CACHE",
                priority=9,
            ),
        ]

    def evaluate(self, stream: EventStream) -> list[tuple[Rule, bool]]:
        results = []
        for rule in sorted(self._rules, key=lambda r: -r.priority):
            fired = rule.evaluate(stream)
            if fired:
                rule.fired_count += 1
            results.append((rule, fired))
        return results

    def top_decision(self, stream: EventStream) -> str | None:
        for rule in sorted(self._rules, key=lambda r: -r.priority):
            if rule.evaluate(stream):
                return rule.action
        return None

    def add_rule(self, rule: Rule) -> None:
        self._rules.append(rule)

    @property
    def rules(self) -> list[Rule]:
        return list(self._rules)
