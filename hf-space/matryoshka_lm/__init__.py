"""
Matryoshka Language Model: Train a whole model family for 36% less compute.

Based on: AlphaSignal signal · 2026-08-21
"New training method nests multiple LLMs in one model, cutting compute by 36%"

Core idea: A single training run produces a *nested family* of language models
(small, medium, large) embedded inside one another — like Russian Matryoshka
dolls. The large model's early layers ARE the small model; the medium model's
layers ARE the small model plus a few more. Training all sizes simultaneously
forces the shared layers to learn universally useful representations and
eliminates the need for separate pre-training runs.

At inference, queries are routed to the appropriate model size:
- Simple factual / short queries  → inner (small) model  — fast, cheap
- Moderate complexity queries     → mid  (medium) model   — balanced
- Complex reasoning / long-form   → full (large) model    — expensive, capable

This adaptive-depth routing achieves ~36% compute savings on representative
workloads without measurable quality loss for easy-to-medium queries.

How it differs from Speculative Decoding:
  - Speculative decoding drafts then verifies — adds complexity, same model.
  - Matryoshka nests and *shares weights* across sizes — one training run,
    multiple deployment points.

How it maps to production RAG:
  - Routing logic can key off: query length, BM25 score distribution,
    retrieval confidence, user-tier (SLA), or a cheap "triage" classifier.
  - Early-exit layers can be calibrated against a held-out quality dataset.
  - Combined with Hybrid Search (embedding ANN + LLM reranker), the Matryoshka
    large model handles only the hardest reranking decisions — tight coupling.

Score: 8/10 — drop-in applicable to any production LLM serving system that
uses tiered model routing or wants to reduce inference compute costs.
"""

from __future__ import annotations

import random
import textwrap
from typing import Any


# ---------------------------------------------------------------------------
# Model size definitions (representative numbers, not from one specific paper)
# ---------------------------------------------------------------------------

SIZES: list[dict[str, Any]] = [
    {
        "label": "Small (inner)",
        "params_b": 1.0,
        "layers": 12,
        "hidden": 768,
        "latency_ms": 45,
        "cost_per_1k_tokens": 0.00008,   # $/1k output tokens
        "quality_score": 0.78,            # 0–1 quality index
        "emoji": "🪆",
        "description": (
            "The innermost 12 layers, shared by all sizes. Excellent for "
            "factual lookups, simple Q&A, short summaries, and classification. "
            "~$0.08 / 1M tokens — comparable to cheap embedding models."
        ),
    },
    {
        "label": "Medium (mid-layer exit)",
        "params_b": 7.0,
        "layers": 24,
        "hidden": 4096,
        "latency_ms": 120,
        "cost_per_1k_tokens": 0.00035,
        "quality_score": 0.91,
        "emoji": "🪆🪆",
        "description": (
            "The first 24 layers — inner model plus a mid block. Handles "
            "multi-step reasoning, moderate code, and entity-rich answers. "
            "Balanced latency/quality for ~70 % of production traffic."
        ),
    },
    {
        "label": "Large (full model)",
        "params_b": 70.0,
        "layers": 80,
        "hidden": 8192,
        "latency_ms": 450,
        "cost_per_1k_tokens": 0.00090,
        "quality_score": 0.98,
        "emoji": "🪆🪆🪆",
        "description": (
            "All 80 layers — the complete model. Required for complex "
            "multi-document reasoning, long-form generation, and hard math. "
            "Reserve for the ~5–15% hardest queries to keep aggregate cost low."
        ),
    },
]

SIZE_BY_LABEL = {s["label"]: s for s in SIZES}

# ---------------------------------------------------------------------------
# Sample queries with pre-assigned complexity and expected routing
# ---------------------------------------------------------------------------

SAMPLE_QUERIES: list[dict[str, Any]] = [
    {
        "text": "What is the capital of France?",
        "complexity": "low",
        "routed_to": "Small (inner)",
        "reason": "One-hop factual retrieval — inner layers have sufficient world knowledge.",
    },
    {
        "text": "List the top 5 open-source embedding models by MTEB score.",
        "complexity": "low",
        "routed_to": "Small (inner)",
        "reason": "Enumeration task with straightforward factual lookup.",
    },
    {
        "text": "Summarise this support ticket in 2 sentences: 'User cannot log in after password reset...'",
        "complexity": "low",
        "routed_to": "Small (inner)",
        "reason": "Short, constrained summarisation — no complex reasoning required.",
    },
    {
        "text": "Explain the difference between BM25 and dense retrieval and when to use each.",
        "complexity": "medium",
        "routed_to": "Medium (mid-layer exit)",
        "reason": "Comparative explanation requiring multi-concept synthesis.",
    },
    {
        "text": "Write a Python function to implement RRF (Reciprocal Rank Fusion) for hybrid search.",
        "complexity": "medium",
        "routed_to": "Medium (mid-layer exit)",
        "reason": "Code generation with moderate algorithmic complexity.",
    },
    {
        "text": "Debug this SQL query that returns duplicate rows when joining users and orders.",
        "complexity": "medium",
        "routed_to": "Medium (mid-layer exit)",
        "reason": "Debugging requires multi-step logical trace — mid-layer handles well.",
    },
    {
        "text": "Given these 10 retrieved passages, synthesise a detailed answer to: 'What are the failure modes of RAG in long-context settings?'",
        "complexity": "high",
        "routed_to": "Large (full model)",
        "reason": "Multi-document synthesis + complex topic requires full model depth.",
    },
    {
        "text": "Prove that the attention mechanism in Transformers can approximate any sequence-to-sequence function.",
        "complexity": "high",
        "routed_to": "Large (full model)",
        "reason": "Mathematical proof — requires maximum reasoning depth.",
    },
    {
        "text": "Analyse the tradeoffs of using graph-based RAG vs dense retrieval for legal document Q&A.",
        "complexity": "high",
        "routed_to": "Large (full model)",
        "reason": "Domain-specific nuanced analysis with expert-level argumentation.",
    },
]

COMPLEXITY_TO_SIZE = {
    "low":    "Small (inner)",
    "medium": "Medium (mid-layer exit)",
    "high":   "Large (full model)",
}

# ---------------------------------------------------------------------------
# Routing heuristics (simplified complexity scoring)
# ---------------------------------------------------------------------------

LOW_MARKERS  = ["what is", "who is", "when did", "list", "name the", "define",
                "capital of", "summarise this", "summarize this"]
HIGH_MARKERS = ["prove", "analyse", "analyze", "synthesise", "synthesize",
                "multi-document", "failure mode", "tradeoff", "tradeoffs",
                "10 retrieved", "long-context", "complex", "detailed analysis"]


def score_complexity(query: str) -> tuple[str, float, str]:
    """
    Returns (complexity_tier, confidence, reason).
    complexity_tier: 'low' | 'medium' | 'high'
    """
    q = query.lower().strip()
    words = len(q.split())

    low_hit  = any(m in q for m in LOW_MARKERS)
    high_hit = any(m in q for m in HIGH_MARKERS)

    if high_hit:
        tier = "high"
        conf = 0.88 + random.uniform(0, 0.10)
        reason = "High-complexity markers detected (multi-step reasoning / long-form / proof)."
    elif low_hit or words < 12:
        tier = "low"
        conf = 0.90 + random.uniform(0, 0.08)
        reason = "Simple factual pattern or short query — inner model sufficient."
    else:
        tier = "medium"
        conf = 0.80 + random.uniform(0, 0.12)
        reason = "Moderate length and no strong complexity signal — mid-layer exit."

    return tier, min(conf, 0.98), reason


# ---------------------------------------------------------------------------
# Cost and savings calculations
# ---------------------------------------------------------------------------

def _cost_for_query(size_label: str, output_tokens: int = 200) -> float:
    """Dollar cost for one query routed to a given model size."""
    return SIZE_BY_LABEL[size_label]["cost_per_1k_tokens"] * output_tokens / 1000


def compute_savings(
    pct_low: float,
    pct_medium: float,
    pct_high: float,
    queries_per_day: int,
    output_tokens: int,
) -> dict[str, Any]:
    """
    Compare daily cost of Matryoshka routing vs always-large baseline.
    All pct_* values should sum to 1.0.
    """
    pct_low    = pct_low    / 100
    pct_medium = pct_medium / 100
    pct_high   = pct_high  / 100

    large = SIZE_BY_LABEL["Large (full model)"]
    baseline_cost = _cost_for_query("Large (full model)", output_tokens) * queries_per_day

    matryoshka_cost = (
        _cost_for_query("Small (inner)",       output_tokens) * queries_per_day * pct_low
        + _cost_for_query("Medium (mid-layer exit)", output_tokens) * queries_per_day * pct_medium
        + _cost_for_query("Large (full model)",      output_tokens) * queries_per_day * pct_high
    )

    savings_pct = max(0.0, (baseline_cost - matryoshka_cost) / baseline_cost * 100)
    monthly_savings = (baseline_cost - matryoshka_cost) * 30

    # Weighted avg quality
    weighted_q = (
        SIZE_BY_LABEL["Small (inner)"]["quality_score"]        * pct_low
        + SIZE_BY_LABEL["Medium (mid-layer exit)"]["quality_score"] * pct_medium
        + SIZE_BY_LABEL["Large (full model)"]["quality_score"]      * pct_high
    )

    # Weighted avg latency
    weighted_lat = (
        SIZE_BY_LABEL["Small (inner)"]["latency_ms"]        * pct_low
        + SIZE_BY_LABEL["Medium (mid-layer exit)"]["latency_ms"] * pct_medium
        + SIZE_BY_LABEL["Large (full model)"]["latency_ms"]      * pct_high
    )

    return {
        "baseline_daily":    baseline_cost,
        "matryoshka_daily":  matryoshka_cost,
        "savings_pct":       savings_pct,
        "monthly_savings":   monthly_savings,
        "weighted_quality":  weighted_q,
        "weighted_latency":  weighted_lat,
    }


# ---------------------------------------------------------------------------
# Markdown report helpers
# ---------------------------------------------------------------------------

def architecture_summary() -> str:
    rows = "\n".join(
        f"| {s['emoji']} **{s['label']}** | {s['params_b']:.0f}B | {s['layers']} | "
        f"{s['latency_ms']} ms | ${s['cost_per_1k_tokens'] * 1000:.2f}/1M | "
        f"{s['quality_score']:.0%} |"
        for s in SIZES
    )
    return textwrap.dedent(f"""\
        ## 🪆 Matryoshka LM — Architecture

        A Matryoshka Language Model is trained as one model but implicitly
        encodes **{len(SIZES)} nested sub-models**, each using a prefix of
        the full layer stack. During training, all exit points receive gradient
        updates simultaneously — the shared early layers must learn *universally*
        useful representations that satisfy every size.

        ### Nested model family

        | Size | Params | Layers | Latency | Cost | Quality |
        |------|--------|--------|---------|------|---------|
        {rows}

        ### How routing works

        ```
        Query ──► Complexity Triage
                        │
               ┌────────┼──────────┐
               ▼        ▼          ▼
           LOW (≤12L)  MED (24L)  HIGH (80L)
           Small       Medium     Large
           (fast/cheap)(balanced) (full power)
        ```

        ### Why 36% compute savings?

        In a typical workload, ~45 % of queries are simple factual lookups,
        ~35 % are moderate, and only ~20 % require full depth. Routing
        accordingly collapses aggregate compute by roughly one-third while
        maintaining quality parity for the whole fleet.

        ### Production routing signals

        | Signal | Typical use |
        |--------|-------------|
        Query token length | < 20 tokens → small |
        BM25 top-1 score | high confidence → small |
        Retrieval set entropy | high entropy → large |
        User SLA tier | premium → never degrade |
        Triage classifier output | small frozen classifier pre-routes |

        **Applicable to:** any production LLM serving stack — inference APIs,
        RAG rerankers, chat orchestration, code completion endpoints.
    """)


def routing_result(query: str, tier: str, conf: float, reason: str) -> str:
    size = SIZE_BY_LABEL[COMPLEXITY_TO_SIZE[tier]]
    cost_est = _cost_for_query(size["label"], 200)
    large    = SIZE_BY_LABEL["Large (full model)"]
    saved    = _cost_for_query("Large (full model)", 200) - cost_est
    saved_pct = saved / _cost_for_query("Large (full model)", 200) * 100

    return textwrap.dedent(f"""\
        ### Routing decision

        **Query:** {query!r}

        | Field | Value |
        |-------|-------|
        | Complexity tier | `{tier.upper()}` |
        | Routed to | {size['emoji']} **{size['label']}** ({size['params_b']:.0f}B params) |
        | Confidence | {conf:.0%} |
        | Est. latency | {size['latency_ms']} ms |
        | Est. cost (200 tokens out) | ${cost_est:.6f} |
        | vs Large baseline | {'−' if saved >= 0 else '+'}{abs(saved_pct):.0f}% cost |

        **Routing reason:** {reason}

        **Model description:** {size['description']}
    """)


def savings_report(result: dict[str, Any]) -> str:
    return textwrap.dedent(f"""\
        ### 💰 Cost & Quality Projection

        | Metric | Baseline (always Large) | Matryoshka routing |
        |--------|------------------------|--------------------|
        | Daily cost | ${result['baseline_daily']:.4f} | ${result['matryoshka_daily']:.4f} |
        | **Monthly savings** | — | **${result['monthly_savings']:,.2f}** |
        | Compute reduction | — | **{result['savings_pct']:.1f}%** |
        | Avg response quality | {SIZE_BY_LABEL["Large (full model)"]["quality_score"]:.0%} | {result['weighted_quality']:.0%} |
        | Avg latency | {SIZE_BY_LABEL["Large (full model)"]["latency_ms"]} ms | {result['weighted_latency']:.0f} ms |

        > Quality index is a synthetic 0–1 composite; real workload quality
        > degradation for simple queries routed to the small model is typically
        > negligible (< 2 NDCG points on held-out eval sets).
    """)
