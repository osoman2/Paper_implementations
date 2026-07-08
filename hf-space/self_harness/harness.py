"""
Harness data structures: system prompt, injected rules, and configuration.

In Self-Harness (Shanghai AI Lab), the harness = system prompt + tool-use logic +
error handling + verification rules. Rules are injected as bullet points into the
system prompt after weakness mining identifies failure patterns.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class HarnessRule:
    """A single injected rule in the agent's system prompt."""
    id: str
    description: str
    trigger_pattern: str
    action: str
    source: str = "manual"  # "manual" | "self_harness" | "harnessX"
    effectiveness: float = 0.0  # estimated pass-rate lift on affected tasks

    def to_prompt_line(self) -> str:
        return f"- [{self.id}] {self.description}: {self.action}"


BUILTIN_RULES: list[HarnessRule] = [
    HarnessRule(
        id="R01",
        description="File existence check",
        trigger_pattern="file.*write|write.*file|overwrite",
        action="Always check if the target file exists before writing. If it exists, confirm before overwriting.",
        source="manual",
        effectiveness=0.85,
    ),
    HarnessRule(
        id="R02",
        description="API retry on timeout",
        trigger_pattern="timeout|api.*call|http.*request",
        action="On timeout or transient error, retry up to 3 times with exponential backoff (2s, 4s, 8s).",
        source="manual",
        effectiveness=0.78,
    ),
    HarnessRule(
        id="R03",
        description="Ambiguity clarification",
        trigger_pattern="unclear|ambiguous|multiple.*interpret",
        action="When an instruction has multiple valid interpretations, ask for clarification before acting.",
        source="manual",
        effectiveness=0.72,
    ),
    HarnessRule(
        id="R04",
        description="Long context chunking",
        trigger_pattern="long.*context|truncat|token.*limit|context.*window",
        action="When input exceeds 4000 tokens, split into overlapping chunks and process sequentially.",
        source="manual",
        effectiveness=0.80,
    ),
    HarnessRule(
        id="R05",
        description="Missing tool fallback",
        trigger_pattern="tool.*not.*found|missing.*tool|unavailable.*function",
        action="When a required tool is unavailable, fall back to the next best available tool and note the limitation.",
        source="manual",
        effectiveness=0.65,
    ),
    HarnessRule(
        id="R06",
        description="Path validation",
        trigger_pattern="path|directory|folder",
        action="Validate all file paths before operations. Create parent directories if missing.",
        source="self_harness",
        effectiveness=0.70,
    ),
    HarnessRule(
        id="R07",
        description="Output format verification",
        trigger_pattern="json|yaml|xml|format",
        action="Always validate structured output against the expected schema before returning.",
        source="self_harness",
        effectiveness=0.75,
    ),
    HarnessRule(
        id="R08",
        description="Rate limit handling",
        trigger_pattern="rate.limit|429|throttl",
        action="On 429 rate-limit responses, wait 60 seconds then retry; surface a user-friendly message.",
        source="harnessX",
        effectiveness=0.82,
    ),
]


class BaseHarness:
    """
    Simulated agent harness: holds the base system prompt + an ordered list
    of injected rules. Each rule is appended as a bullet point to the system prompt.
    """

    BASE_SYSTEM_PROMPT = (
        "You are an AI assistant that executes tasks autonomously. "
        "Complete each task carefully and return a structured result."
    )

    def __init__(self, active_rule_ids: list[str] | None = None):
        self.all_rules: dict[str, HarnessRule] = {r.id: r for r in BUILTIN_RULES}
        self.active_ids: list[str] = active_rule_ids or []

    def add_rule(self, rule: HarnessRule) -> None:
        self.all_rules[rule.id] = rule
        if rule.id not in self.active_ids:
            self.active_ids.append(rule.id)

    def remove_rule(self, rule_id: str) -> None:
        if rule_id in self.active_ids:
            self.active_ids.remove(rule_id)

    @property
    def active_rules(self) -> list[HarnessRule]:
        return [self.all_rules[rid] for rid in self.active_ids if rid in self.all_rules]

    def build_system_prompt(self) -> str:
        prompt = self.BASE_SYSTEM_PROMPT
        rules = self.active_rules
        if rules:
            rule_lines = "\n".join(r.to_prompt_line() for r in rules)
            prompt += f"\n\nOperating rules:\n{rule_lines}"
        return prompt

    def covers(self, failure_pattern: str) -> bool:
        """Returns True if any active rule targets this failure pattern."""
        import re
        for rule in self.active_rules:
            if re.search(rule.trigger_pattern, failure_pattern, re.IGNORECASE):
                return True
        return False

    def clone_with_rules(self, extra_rule_ids: list[str]) -> "BaseHarness":
        new_harness = BaseHarness(active_rule_ids=list(self.active_ids))
        new_harness.all_rules = dict(self.all_rules)
        for rid in extra_rule_ids:
            if rid not in new_harness.active_ids:
                new_harness.active_ids.append(rid)
        return new_harness

    def summary(self) -> str:
        rules = self.active_rules
        if not rules:
            return "No rules active (base harness only)"
        lines = [f"  • {r.id}: {r.description}" for r in rules]
        return f"{len(rules)} active rules:\n" + "\n".join(lines)
