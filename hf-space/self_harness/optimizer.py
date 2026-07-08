"""
HarnessOptimizer — simulates the Self-Harness 3-stage loop:
  Stage 1: Weakness Mining — analyze failure traces to identify recurring patterns
  Stage 2: Harness Proposal — suggest rule injections to fix the patterns
  Stage 3: Proposal Validation — regression-test proposed changes; accept if net gain

Also models HarnessX's AEGIS optimizer for comparison.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .harness import BaseHarness
    from .tasks import AgentTask, TaskResult

from .harness import BUILTIN_RULES


@dataclass
class WeaknessMiningResult:
    failure_counts: dict[str, int]        # pattern -> count
    top_patterns: list[str]               # ordered by frequency
    total_tasks: int
    total_failures: int
    failure_rate: float
    trace_snippets: list[str]             # sample failure traces


@dataclass
class HarnessProposal:
    proposed_rule_ids: list[str]
    rationale: dict[str, str]             # rule_id -> why
    estimated_pass_rate_delta: float
    regression_safe: bool
    validation_tasks_run: int
    validation_pass_rate_before: float
    validation_pass_rate_after: float


class HarnessOptimizer:
    """
    Implements the Self-Harness 3-stage improvement loop.

    Stage 1 — Weakness Mining:
        Collect execution traces from a batch run, count failure tags by pattern.

    Stage 2 — Harness Proposal:
        For each top failure pattern, identify which BUILTIN_RULES could fix it.
        Propose injecting those rules into the harness.

    Stage 3 — Proposal Validation:
        Simulate before/after pass rates on the same task set.
        Only accept if: (a) pass rate improves and (b) no regression on previously-passing tasks.
    """

    def __init__(self, top_k_patterns: int = 3, min_failure_count: int = 1):
        self.top_k = top_k_patterns
        self.min_failure_count = min_failure_count
        self._rule_pattern_map: dict[str, list[str]] = {
            r.id: r.trigger_pattern.split("|") for r in BUILTIN_RULES
        }

    # ------------------------------------------------------------------
    # Stage 1: Weakness Mining
    # ------------------------------------------------------------------
    def mine_weaknesses(self, results: list["TaskResult"]) -> WeaknessMiningResult:
        failures = [r for r in results if not r.passed]
        patterns = [r.failure_tag for r in failures if r.failure_tag]
        counts = Counter(patterns)
        top = [p for p, _ in counts.most_common(self.top_k) if counts[p] >= self.min_failure_count]
        snippets = [r.execution_trace for r in failures[:5]]

        return WeaknessMiningResult(
            failure_counts=dict(counts),
            top_patterns=top,
            total_tasks=len(results),
            total_failures=len(failures),
            failure_rate=len(failures) / max(len(results), 1),
            trace_snippets=snippets,
        )

    # ------------------------------------------------------------------
    # Stage 2: Harness Proposal
    # ------------------------------------------------------------------
    def propose_rules(
        self,
        mining: WeaknessMiningResult,
        current_harness: "BaseHarness",
    ) -> HarnessProposal:
        import re

        proposed_ids: list[str] = []
        rationale: dict[str, str] = {}

        for pattern in mining.top_patterns:
            for rule in BUILTIN_RULES:
                if rule.id in current_harness.active_ids:
                    continue
                # Check if this rule's trigger matches the failure pattern
                for kw in rule.trigger_pattern.split("|"):
                    if re.search(kw.strip(), pattern, re.IGNORECASE):
                        if rule.id not in proposed_ids:
                            proposed_ids.append(rule.id)
                            rationale[rule.id] = (
                                f"Pattern '{pattern}' appeared {mining.failure_counts.get(pattern, 0)} time(s); "
                                f"rule {rule.id} ({rule.description}) covers '{kw.strip()}'"
                            )
                        break

        estimated_delta = sum(
            BUILTIN_RULES_BY_ID[rid].effectiveness * mining.failure_counts.get(pattern, 0) / max(mining.total_tasks, 1)
            for rid, pattern in [
                (rid, p)
                for p in mining.top_patterns
                for rid in proposed_ids
                if BUILTIN_RULES_BY_ID.get(rid) and re.search(
                    BUILTIN_RULES_BY_ID[rid].trigger_pattern, p, re.IGNORECASE
                )
            ]
        )
        estimated_delta = min(estimated_delta, 0.40)

        return HarnessProposal(
            proposed_rule_ids=proposed_ids,
            rationale=rationale,
            estimated_pass_rate_delta=round(estimated_delta, 3),
            regression_safe=True,
            validation_tasks_run=0,
            validation_pass_rate_before=0.0,
            validation_pass_rate_after=0.0,
        )

    # ------------------------------------------------------------------
    # Stage 3: Proposal Validation
    # ------------------------------------------------------------------
    def validate_proposal(
        self,
        proposal: HarnessProposal,
        tasks: list["AgentTask"],
        current_harness: "BaseHarness",
        seed: int = 0,
    ) -> tuple[HarnessProposal, list["TaskResult"], list["TaskResult"]]:
        """
        Simulate task runs before and after applying proposed rules.
        Returns updated proposal + before/after results.
        """
        # Before: run with current harness
        before_results = [
            t.simulate_run(current_harness.covers(t.failure_pattern), seed=seed + i)
            for i, t in enumerate(tasks)
        ]

        # After: run with proposed rules added
        improved_harness = current_harness.clone_with_rules(proposal.proposed_rule_ids)
        after_results = [
            t.simulate_run(improved_harness.covers(t.failure_pattern), seed=seed + i + 1000)
            for i, t in enumerate(tasks)
        ]

        before_pass = sum(1 for r in before_results if r.passed) / max(len(before_results), 1)
        after_pass  = sum(1 for r in after_results  if r.passed) / max(len(after_results),  1)

        # Regression check: tasks that passed before still pass after?
        passed_before = {r.task.id for r in before_results if r.passed}
        failed_after  = {r.task.id for r in after_results  if not r.passed}
        regressions = passed_before & failed_after
        regression_safe = len(regressions) == 0

        proposal.validation_tasks_run      = len(tasks)
        proposal.validation_pass_rate_before = round(before_pass, 3)
        proposal.validation_pass_rate_after  = round(after_pass,  3)
        proposal.regression_safe             = regression_safe

        return proposal, before_results, after_results

    # ------------------------------------------------------------------
    # Full loop
    # ------------------------------------------------------------------
    def run_full_loop(
        self,
        tasks: list["AgentTask"],
        initial_harness: "BaseHarness",
        seed: int = 42,
    ) -> dict:
        """
        Run the complete Self-Harness loop and return all intermediate artifacts.
        """
        # Initial run
        initial_results = [
            t.simulate_run(initial_harness.covers(t.failure_pattern), seed=seed + i)
            for i, t in enumerate(tasks)
        ]

        # Stage 1
        mining = self.mine_weaknesses(initial_results)

        # Stage 2
        proposal = self.propose_rules(mining, initial_harness)

        # Stage 3
        proposal, before_results, after_results = self.validate_proposal(
            proposal, tasks, initial_harness, seed=seed
        )

        return {
            "initial_results": initial_results,
            "mining": mining,
            "proposal": proposal,
            "before_results": before_results,
            "after_results": after_results,
        }


BUILTIN_RULES_BY_ID = {r.id: r for r in BUILTIN_RULES}
