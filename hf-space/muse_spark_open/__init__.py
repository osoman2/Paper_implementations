"""
Muse Spark Open — Meta's 816B MoE Model POC (Tab 51)
Source: AlphaSignal Top News · 2026-10-16 · news@alphasignal.ai
Score: 8/10

Meta's Muse Spark Open is an 816B parameter Mixture-of-Experts model with:
  - Only 42B parameters active per request (256 experts, top-8 routing)
  - 1M token context window (entire codebases in one call)
  - Native multimodal: text, images, video, audio
  - Local inference on two M5 Ultra Mac Studios via MLX
  - Cloud: Groq (low latency), Together AI, OpenRouter
"""

from .simulator import (
    ARCHITECTURE_CARDS,
    BENCHMARK_ROWS,
    IMPL_PATTERNS,
    benchmark_table_html,
    compute_rag_cost,
)

__all__ = [
    "ARCHITECTURE_CARDS",
    "BENCHMARK_ROWS",
    "IMPL_PATTERNS",
    "benchmark_table_html",
    "compute_rag_cost",
]
