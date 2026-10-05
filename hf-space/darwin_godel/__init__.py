"""
Darwin Gödel Machine & Self-Evolving Agent Stack
Based on AlphaSignal Deep Dive (2026-10-04):
  "How AI Agents Are Learning to Rewrite Their Own Stack"

Covers: SkillOpt, WikiSkill, Self-Harness, Darwin Gödel Machine (DGM),
        Hyperagents (Meta), HarnessX (Xiaomi), EnvHarness, Raven (EverMind AI)
"""

from .dgm import (
    STACK_LAYERS,
    DGMVariant,
    DGMArchive,
    run_dgm_exploration,
    run_skill_opt,
    run_env_harness,
    IMPL_PATTERNS,
    BENCHMARK_DATA,
)

__all__ = [
    "STACK_LAYERS",
    "DGMVariant",
    "DGMArchive",
    "run_dgm_exploration",
    "run_skill_opt",
    "run_env_harness",
    "IMPL_PATTERNS",
    "BENCHMARK_DATA",
]
