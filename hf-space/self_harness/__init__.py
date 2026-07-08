"""
Self-Harness & HarnessX — Self-Improving AI Agent Harness
Based on AlphaSignal deep-dive (2026-07-05), covering:
  - Self-Harness (Shanghai AI Lab): 3-stage loop for autonomous harness evolution
  - HarnessX (Xiaomi Darwin Agent Team): processor-based agent foundry with AEGIS RL optimizer
"""

from .harness import BaseHarness, HarnessRule
from .tasks import TASK_LIBRARY, AgentTask, TaskResult
from .optimizer import HarnessOptimizer, WeaknessMiningResult, HarnessProposal

__all__ = [
    "BaseHarness",
    "HarnessRule",
    "TASK_LIBRARY",
    "AgentTask",
    "TaskResult",
    "HarnessOptimizer",
    "WeaknessMiningResult",
    "HarnessProposal",
]
