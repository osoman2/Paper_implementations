"""
Fork-and-Flush for Research Agents — POC (Tab 50)
Source: AlphaSignal newsletter · 2026-10-15 · news@alphasignal.ai
Score: 9/10

Simple technique that boosts research agent performance by 66%:
  - Fork: spawn N parallel copies of the agent from a shared checkpoint
  - Flush: wipe each fork's memory (tool-call history / accumulated context) independently
  - Merge: collect results and pick the best / combine non-overlapping discoveries

Escapes "idea ruts" by preventing each fork from reinforcing the same exploration bias.
"""

from .simulator import (
    TECHNIQUE_CARDS,
    BENCHMARK_ROWS,
    IMPL_PATTERNS,
    benchmark_table_html,
    run_fork_flush_simulation,
)

__all__ = [
    "TECHNIQUE_CARDS",
    "BENCHMARK_ROWS",
    "IMPL_PATTERNS",
    "benchmark_table_html",
    "run_fork_flush_simulation",
]
