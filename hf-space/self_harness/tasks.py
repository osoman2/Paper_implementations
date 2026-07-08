"""
Task library with known failure patterns used to demo the Self-Harness loop.

Each task has:
  - description: natural language task for the agent
  - failure_pattern: the category of failure this task triggers without a matching rule
  - required_rule_ids: which harness rules prevent this failure
  - base_pass_rate: probability of passing WITHOUT the required rules
  - ruled_pass_rate: probability of passing WITH the required rules
"""
from __future__ import annotations
from dataclasses import dataclass
import random


@dataclass
class AgentTask:
    id: str
    description: str
    failure_pattern: str
    required_rule_ids: list[str]
    base_pass_rate: float      # without any protective rule
    ruled_pass_rate: float     # with the right rule(s) active
    category: str

    def simulate_run(self, harness_covers: bool, seed: int | None = None) -> "TaskResult":
        rng = random.Random(seed)
        rate = self.ruled_pass_rate if harness_covers else self.base_pass_rate
        passed = rng.random() < rate
        trace = self._build_trace(passed, harness_covers)
        return TaskResult(
            task=self,
            passed=passed,
            harness_covered=harness_covers,
            execution_trace=trace,
            failure_tag=None if passed else self.failure_pattern,
        )

    def _build_trace(self, passed: bool, covered: bool) -> str:
        if passed:
            return f"[OK] Task '{self.id}' completed successfully."
        prefix = "[RULE_PREVENTED]" if covered else "[FAIL]"
        messages = {
            "file_overwrite": f"{prefix} FileExistsError — attempted to write to existing output.log without checking",
            "api_timeout": f"{prefix} TimeoutError — API call exceeded 30s limit, no retry attempted",
            "ambiguous_instruction": f"{prefix} ValueError — could not determine intent: multiple valid interpretations found",
            "context_overflow": f"{prefix} TokenLimitError — input exceeded model context window (4096 tokens), truncated",
            "missing_tool": f"{prefix} ToolNotFoundError — required tool 'web_search' not available in this environment",
            "invalid_path": f"{prefix} FileNotFoundError — parent directory '/data/output/2026/' does not exist",
            "malformed_output": f"{prefix} ValidationError — returned JSON missing required field 'result_id'",
            "rate_limit": f"{prefix} HTTPError 429 — API rate limit exceeded, no backoff applied",
        }
        return messages.get(self.failure_pattern, f"{prefix} UnknownError in task '{self.id}'")


@dataclass
class TaskResult:
    task: AgentTask
    passed: bool
    harness_covered: bool
    execution_trace: str
    failure_tag: str | None


TASK_LIBRARY: list[AgentTask] = [
    # File overwrite failures (R01)
    AgentTask("T01", "Save analysis results to output.log", "file_overwrite", ["R01"], 0.30, 0.92, "file_ops"),
    AgentTask("T02", "Write processed data to results/report.json", "file_overwrite", ["R01"], 0.25, 0.90, "file_ops"),
    AgentTask("T03", "Export embeddings to cache/vectors.pkl", "file_overwrite", ["R01", "R06"], 0.20, 0.88, "file_ops"),
    # API timeout failures (R02)
    AgentTask("T04", "Fetch document metadata via REST API", "api_timeout", ["R02"], 0.40, 0.88, "api"),
    AgentTask("T05", "Call embedding service for batch of 100 texts", "api_timeout", ["R02"], 0.35, 0.85, "api"),
    AgentTask("T06", "Query external knowledge base endpoint", "api_timeout", ["R02"], 0.45, 0.87, "api"),
    # Ambiguous instructions (R03)
    AgentTask("T07", "Update the configuration for the production system", "ambiguous_instruction", ["R03"], 0.50, 0.85, "planning"),
    AgentTask("T08", "Summarize the important documents", "ambiguous_instruction", ["R03"], 0.45, 0.82, "planning"),
    # Context overflow (R04)
    AgentTask("T09", "Summarize the 300-page technical specification", "context_overflow", ["R04"], 0.15, 0.87, "llm"),
    AgentTask("T10", "Extract entities from 50 combined research papers", "context_overflow", ["R04"], 0.10, 0.83, "llm"),
    # Missing tool fallback (R05)
    AgentTask("T11", "Search the web for latest RAG benchmarks", "missing_tool", ["R05"], 0.20, 0.75, "tools"),
    AgentTask("T12", "Run code interpreter on the analysis script", "missing_tool", ["R05"], 0.25, 0.70, "tools"),
    # Path/directory failures (R06)
    AgentTask("T13", "Create dataset index in /data/output/2026/q2/", "invalid_path", ["R06"], 0.35, 0.90, "file_ops"),
    # Output format failures (R07)
    AgentTask("T14", "Return search results as JSON with required schema", "malformed_output", ["R07"], 0.55, 0.92, "output"),
    AgentTask("T15", "Generate structured evaluation report", "malformed_output", ["R07"], 0.50, 0.89, "output"),
    # Rate limit failures (R08)
    AgentTask("T16", "Process 1000 documents via OpenAI batch API", "rate_limit", ["R08"], 0.30, 0.88, "api"),
    AgentTask("T17", "Reindex vector database with 10k embeddings", "rate_limit", ["R08"], 0.35, 0.85, "api"),
    # Mixed / harder tasks
    AgentTask("T18", "Backup and re-index the production knowledge base", "file_overwrite", ["R01", "R02"], 0.15, 0.82, "complex"),
    AgentTask("T19", "Migrate embedding model outputs to new schema", "malformed_output", ["R06", "R07"], 0.20, 0.85, "complex"),
    AgentTask("T20", "Run full RAG pipeline evaluation on 500 queries", "api_timeout", ["R02", "R08"], 0.25, 0.87, "complex"),
]

TASK_CATEGORIES = sorted(set(t.category for t in TASK_LIBRARY))
TASK_BY_ID = {t.id: t for t in TASK_LIBRARY}
