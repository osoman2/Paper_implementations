"""
KV Cache Techniques for Multi-Model Agents — POC (Tab 49)
Source: AlphaSignal Deep Dive, 2026-10-12 · news@alphasignal.ai
Score: 9/10

Covers two complementary handoff-acceleration techniques:
  1. Nvidia cross-model KV cache transfer  — 25× speedup (model switching)
  2. Cache-2-Cache (C2C)                   — 2.5× avg speedup (specialist delegation)
"""

from .kv_transfer import (
    TECHNIQUES,
    BENCHMARK_ROWS,
    IMPL_PATTERNS,
    benchmark_table_html,
    run_handoff_simulation,
    simulate_handoff,
)

__all__ = [
    "TECHNIQUES",
    "BENCHMARK_ROWS",
    "IMPL_PATTERNS",
    "benchmark_table_html",
    "run_handoff_simulation",
    "simulate_handoff",
]
