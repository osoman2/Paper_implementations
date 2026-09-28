"""
System One Models: Fast Typed AI Decisions for Production Stacks
Source: AlphaSignal Deep Dive · 2026-09-27 · news@alphasignal.ai
Covers: Jev (TypeSafe AI), Laya (open-source), CLM-8B (Stanford/Nvidia)
Score: 9/10 — New model category for fast routing, classification, scoring in agent pipelines
"""

import math

# ── Latency / cost comparison data ────────────────────────────────────────────

BENCHMARK_DATA = [
    {
        "model": "GPT-4o",
        "category": "Frontier LLM",
        "latency_ms": 2800,
        "cost_per_m_input": 2.50,
        "cost_per_m_output": 10.00,
        "task_type": "decision",
        "accuracy_routing": 94.2,
        "note": "Autoregressive; metered output tokens",
    },
    {
        "model": "Claude Haiku 4.5",
        "category": "Fast LLM",
        "latency_ms": 600,
        "cost_per_m_input": 0.80,
        "cost_per_m_output": 4.00,
        "task_type": "decision",
        "accuracy_routing": 92.1,
        "note": "Autoregressive; fastest LLM option",
    },
    {
        "model": "Jev (TypeSafe AI)",
        "category": "System One",
        "latency_ms": 120,
        "cost_per_m_input": 0.042,
        "cost_per_m_output": 0.0,
        "task_type": "decision",
        "accuracy_routing": 93.8,
        "note": "Parallel output; no output token cost; RLCD-trained",
    },
    {
        "model": "Laya (421M)",
        "category": "System One (open)",
        "latency_ms": 33,
        "cost_per_m_input": 0.0,
        "cost_per_m_output": 0.0,
        "task_type": "decision",
        "accuracy_routing": 89.5,
        "note": "Self-hosted ModernBERT; Tesla T4 inference",
    },
    {
        "model": "CLM-8B (Stanford/Nvidia)",
        "category": "System One (research)",
        "latency_ms": 14,
        "cost_per_m_input": 0.0,
        "cost_per_m_output": 0.0,
        "task_type": "decision",
        "accuracy_routing": 91.2,
        "note": "Contrastive embedding; 9x faster than Jev on large action sets",
    },
]

# ── Decision schema examples ───────────────────────────────────────────────────

DECISION_SCHEMAS = [
    {
        "use_case": "Support Ticket Routing",
        "input_description": "Customer support message text + metadata",
        "schema": {
            "department": ["billing", "technical", "sales", "fraud", "general"],
            "urgency": ["critical", "high", "medium", "low"],
            "sentiment": ["frustrated", "neutral", "satisfied"],
            "requires_human": [True, False],
        },
        "example_input": "My card was charged twice for the same order and I need this fixed today!",
        "expected_output": {
            "department": "billing",
            "urgency": "high",
            "sentiment": "frustrated",
            "requires_human": True,
        },
        "llm_equivalent_tokens": 150,
        "system_one_tokens": 12,
    },
    {
        "use_case": "Agent Tool Selection",
        "input_description": "Current agent state + available tools + task description",
        "schema": {
            "selected_tool": ["web_search", "code_exec", "file_read", "calculator", "database_query", "none"],
            "confidence": ["high", "medium", "low"],
            "fallback_to_llm": [True, False],
        },
        "example_input": "Task: find the latest exchange rate for USD/EUR. Tools available: web_search, calculator, database_query",
        "expected_output": {
            "selected_tool": "web_search",
            "confidence": "high",
            "fallback_to_llm": False,
        },
        "llm_equivalent_tokens": 200,
        "system_one_tokens": 18,
    },
    {
        "use_case": "RAG Answer Reranking",
        "input_description": "Query + 5 candidate answers + retrieved passages",
        "schema": {
            "best_answer_index": [0, 1, 2, 3, 4],
            "quality": ["excellent", "good", "acceptable", "poor"],
            "needs_regeneration": [True, False],
            "hallucination_risk": ["none", "low", "medium", "high"],
        },
        "example_input": "Q: What is the capital of France? Candidates: [Paris, Lyon, Marseille, Bordeaux, Nice]",
        "expected_output": {
            "best_answer_index": 0,
            "quality": "excellent",
            "needs_regeneration": False,
            "hallucination_risk": "none",
        },
        "llm_equivalent_tokens": 300,
        "system_one_tokens": 20,
    },
    {
        "use_case": "Content Moderation",
        "input_description": "User-generated text content",
        "schema": {
            "verdict": ["safe", "review", "remove"],
            "category": ["safe", "spam", "harassment", "misinformation", "explicit"],
            "confidence_score": ["high", "medium", "low"],
            "auto_action": ["approve", "queue_review", "reject"],
        },
        "example_input": "Check out this amazing deal! Click here NOW to win $10,000!!!",
        "expected_output": {
            "verdict": "review",
            "category": "spam",
            "confidence_score": "high",
            "auto_action": "queue_review",
        },
        "llm_equivalent_tokens": 180,
        "system_one_tokens": 14,
    },
]

# ── Routing demo examples ──────────────────────────────────────────────────────

ROUTING_EXAMPLES = [
    {
        "text": "I've been charged twice for my subscription this month. Please refund ASAP.",
        "schema": "Support Ticket Routing",
        "result": {"department": "billing", "urgency": "high", "sentiment": "frustrated", "requires_human": True},
        "confidence": 0.97,
        "latency_jev_ms": 89,
        "latency_llm_ms": 2340,
        "cost_jev_usd": 0.000001,
        "cost_llm_usd": 0.00038,
    },
    {
        "text": "Just wondering if you offer student discounts? No rush!",
        "schema": "Support Ticket Routing",
        "result": {"department": "sales", "urgency": "low", "sentiment": "neutral", "requires_human": False},
        "confidence": 0.99,
        "latency_jev_ms": 72,
        "latency_llm_ms": 1980,
        "cost_jev_usd": 0.0000008,
        "cost_llm_usd": 0.00031,
    },
    {
        "text": "My API key keeps returning 401 errors even after rotation. Blocking production deploy.",
        "schema": "Support Ticket Routing",
        "result": {"department": "technical", "urgency": "critical", "sentiment": "frustrated", "requires_human": True},
        "confidence": 0.95,
        "latency_jev_ms": 104,
        "latency_llm_ms": 2710,
        "cost_jev_usd": 0.0000012,
        "cost_llm_usd": 0.00043,
    },
    {
        "text": "Task: summarise last month's sales report. Available: file_read, database_query, web_search",
        "schema": "Agent Tool Selection",
        "result": {"selected_tool": "database_query", "confidence": "high", "fallback_to_llm": False},
        "confidence": 0.93,
        "latency_jev_ms": 95,
        "latency_llm_ms": 2550,
        "cost_jev_usd": 0.0000011,
        "cost_llm_usd": 0.00041,
    },
    {
        "text": "BUY NOW!!! Cheap meds, no prescription needed. Limited time offer click here!!!",
        "schema": "Content Moderation",
        "result": {"verdict": "remove", "category": "spam", "confidence_score": "high", "auto_action": "reject"},
        "confidence": 0.999,
        "latency_jev_ms": 61,
        "latency_llm_ms": 2100,
        "cost_jev_usd": 0.00000072,
        "cost_llm_usd": 0.00033,
    },
]

# ── Implementation patterns ────────────────────────────────────────────────────

IMPL_PATTERNS = [
    {
        "name": "Pattern 1: Jev-as-Judge (Replace LLM Evaluator)",
        "description": (
            "Use a System One model to rank/select between multiple LLM-generated candidates. "
            "Expensive reasoning LLM generates options; cheap decision model picks the best. "
            "Typically 40–200× cheaper evaluation loop with similar or better accuracy."
        ),
        "code": '''\
from typesafe_ai import Jev

jev = Jev(api_key="...")

def jev_judge(candidates: list[str], question: str) -> dict:
    """Pick the best candidate answer using Jev instead of LLM-as-judge."""
    state = {
        "question": question,
        "candidates": candidates,
    }
    schema = {
        "best_index": list(range(len(candidates))),
        "quality": ["excellent", "good", "acceptable", "poor"],
        "hallucination_risk": ["none", "low", "medium", "high"],
    }
    result = jev.decide(state=state, schema=schema)
    # result.best_index → int, result.confidence → float
    return {
        "winner": candidates[result.best_index],
        "quality": result.quality,
        "confidence": result.confidence,
        "cost_usd": result.cost_usd,  # typically ~$0.000001
    }

# Usage: generate candidates with GPT-4 / Claude, rank with Jev
candidates = llm.generate_n(question, n=5)
winner = jev_judge(candidates, question)
''',
    },
    {
        "name": "Pattern 2: System One Router (Cascade Decision Gate)",
        "description": (
            "Route requests through a fast System One model first. High-confidence simple "
            "decisions resolve immediately (~100ms); ambiguous ones escalate to a full LLM. "
            "In production, ~70% of routing tasks never reach the LLM, cutting costs by 10–100×."
        ),
        "code": '''\
from typesafe_ai import Jev
from anthropic import Anthropic

jev = Jev(api_key="...")
claude = Anthropic()

ROUTE_SCHEMA = {
    "department": ["billing", "technical", "sales", "fraud", "general"],
    "urgency": ["critical", "high", "medium", "low"],
    "requires_human": [True, False],
}

def smart_router(ticket_text: str) -> dict:
    """Route with Jev; escalate to LLM only on low confidence."""
    result = jev.decide(state={"text": ticket_text}, schema=ROUTE_SCHEMA)

    if result.confidence < 0.75:
        # Escalate uncertain cases to LLM
        llm_result = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            messages=[{
                "role": "user",
                "content": f"Classify this ticket: {ticket_text}\\n"
                           f"Reply with JSON matching: {ROUTE_SCHEMA}"
            }]
        )
        return {"source": "llm", "raw": llm_result.content[0].text}

    return {
        "source": "jev",
        "department": result.department,
        "urgency": result.urgency,
        "requires_human": result.requires_human,
        "confidence": result.confidence,
        "latency_ms": result.latency_ms,
    }
''',
    },
    {
        "name": "Pattern 3: RLCD Uncertainty Thresholding",
        "description": (
            "Jev is trained with Reinforcement Learning for Calibrated Decisions (RLCD). "
            "Its confidence scores are well-calibrated: 0.98 really means 98% correct. "
            "Use thresholds to build tiered workflows: auto-approve high confidence, "
            "queue medium, send low to human review."
        ),
        "code": '''\
from typesafe_ai import Jev

jev = Jev(api_key="...")

MODERATION_SCHEMA = {
    "verdict": ["safe", "review", "remove"],
    "category": ["safe", "spam", "harassment", "misinformation", "explicit"],
}

def tiered_moderation(content: str) -> dict:
    result = jev.decide(state={"content": content}, schema=MODERATION_SCHEMA)
    conf = result.confidence

    if conf >= 0.95:
        # Auto-act: high confidence, no human needed
        action = "auto_" + ("approve" if result.verdict == "safe" else "reject")
    elif conf >= 0.70:
        # Queue for fast human review
        action = "queue_human_review"
    else:
        # Low confidence: flag for expert review
        action = "escalate_expert"

    return {
        "verdict": result.verdict,
        "category": result.category,
        "confidence": conf,
        "action": action,
        "cost_usd": result.cost_usd,
    }
''',
    },
    {
        "name": "Pattern 4: Open-Source (Laya / CLM) Self-Hosted Decision Layer",
        "description": (
            "For fully self-hosted or air-gapped environments, run Laya (421M params, "
            "33ms on T4) or CLM-8B (14ms for large action sets). Both support the same "
            "state + typed-schema interface as Jev, with zero API costs."
        ),
        "code": '''\
# Laya: open-source Jev-compatible alternative
from laya import Laya

laya = Laya(model="laya-multilingual-v1", device="cuda")  # or "cpu"

TOOL_SCHEMA = {
    "tool": ["web_search", "code_exec", "file_read", "calculator", "none"],
    "confidence": ["high", "medium", "low"],
}

def select_tool_laya(agent_state: dict, available_tools: list[str]) -> dict:
    result = laya.decide(
        state={**agent_state, "available_tools": available_tools},
        schema={"tool": available_tools + ["none"], "confidence": ["high", "medium", "low"]},
    )
    return {"tool": result.tool, "confidence": result.confidence, "latency_ms": result.latency_ms}


# CLM: contrastive embedding approach (Stanford/Nvidia)
from clm import CLM

clm = CLM.from_pretrained("stanford-nvidia/clm-8b")

def rank_actions_clm(state_embedding, action_embeddings) -> int:
    """Return index of best action using contrastive matching."""
    scores = clm.score(state_embedding, action_embeddings)
    return scores.argmax().item()
''',
    },
]

# ── HTML generators ────────────────────────────────────────────────────────────

_CSS = """
<style>
:root {
  --bg: #0f1117; --bg2: #1a1d27; --bg3: #242736;
  --accent: #7c6af7; --accent2: #4ecdc4; --ok: #4caf82;
  --warn: #f5a623; --text: #e8eaf0; --muted: #8b8fa8;
  --border: #2e3148; --radius: 10px; --font: 'Inter', system-ui, sans-serif;
}
body, .gr-prose { background: var(--bg); color: var(--text); font-family: var(--font); }
.card { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius);
        padding: 18px 22px; margin: 10px 0; }
.pill { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 12px;
        font-weight: 600; margin: 2px; }
.pill-llm { background: #3d2b1a; color: #f5a623; border: 1px solid #5a3e22; }
.pill-s1  { background: #1a2d3a; color: #4ecdc4; border: 1px solid #224455; }
.pill-open{ background: #1a2b1a; color: #4caf82; border: 1px solid #224422; }
.kpi-row  { display: flex; flex-wrap: wrap; gap: 14px; margin: 16px 0; }
.kpi { background: var(--bg3); border: 1px solid var(--border); border-radius: 8px;
       padding: 14px 20px; flex: 1; min-width: 140px; text-align: center; }
.kpi .val { font-size: 2em; font-weight: 700; color: var(--accent); }
.kpi .lbl { font-size: 12px; color: var(--muted); margin-top: 4px; }
.bar-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.bar-label { width: 200px; font-size: 13px; color: var(--text); text-align: right; }
.bar-wrap { flex: 1; background: var(--bg3); border-radius: 4px; height: 22px; overflow: hidden; }
.bar { height: 100%; border-radius: 4px; display: flex; align-items: center;
       padding-left: 8px; font-size: 12px; font-weight: 600; color: #fff; }
.bar-llm  { background: linear-gradient(90deg, #8b5e15, #f5a623); }
.bar-s1   { background: linear-gradient(90deg, #1a5c5a, #4ecdc4); }
.bar-open { background: linear-gradient(90deg, #1a4a1a, #4caf82); }
table.cmp { width: 100%; border-collapse: collapse; font-size: 13px; }
table.cmp th { background: var(--bg3); color: var(--muted); font-weight: 600;
               padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }
table.cmp td { padding: 8px 12px; border-bottom: 1px solid var(--border); }
table.cmp tr:hover td { background: var(--bg3); }
.arch-box { border: 1px solid var(--accent); border-radius: 8px; padding: 12px 16px;
            background: var(--bg3); margin: 8px 0; }
.arch-title { color: var(--accent); font-weight: 700; margin-bottom: 6px; font-size: 14px; }
h2, h3 { color: var(--text); }
h2 { font-size: 1.4em; margin-bottom: 8px; }
h3 { font-size: 1.1em; margin-top: 16px; margin-bottom: 6px; color: var(--accent); }
.insight { background: var(--bg3); border-left: 3px solid var(--accent2);
           padding: 10px 16px; margin: 12px 0; border-radius: 4px; font-size: 13px; }
.schema-field { background: var(--bg3); border: 1px solid var(--border); border-radius: 6px;
                padding: 8px 14px; margin: 4px 0; display: flex; align-items: flex-start; gap: 12px; }
.schema-key { font-weight: 700; color: var(--accent); min-width: 160px; font-size: 13px; }
.schema-values { color: var(--muted); font-size: 12px; }
.tag { display: inline-block; background: var(--bg); border: 1px solid var(--border);
       border-radius: 4px; padding: 2px 7px; margin: 2px; font-family: monospace; font-size: 11px; }
.result-box { background: var(--bg2); border: 1px solid var(--ok); border-radius: 8px;
              padding: 14px 18px; margin-top: 12px; }
.result-row { display: flex; justify-content: space-between; padding: 4px 0;
              border-bottom: 1px solid var(--border); font-size: 13px; }
.result-key { color: var(--muted); }
.result-val { font-weight: 600; color: var(--ok); }
.conf-bar { background: var(--bg3); border-radius: 4px; height: 10px; width: 100%; margin-top: 6px; }
.conf-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, var(--ok), var(--accent2)); }
</style>
"""


def architecture_html() -> str:
    return _CSS + """
<div style="max-width:860px;margin:0 auto">
<h2>🏗️ System One Models: A New Layer in the AI Stack</h2>
<p style="color:var(--muted);font-size:14px">
Source: AlphaSignal Deep Dive · 2026-09-27 · <strong>Jev</strong> by TypeSafe AI,
<strong>Laya</strong> (open-source), <strong>CLM-8B</strong> (Stanford/Nvidia) — Score 9/10
</p>

<div class="kpi-row">
  <div class="kpi"><div class="val">444×</div><div class="lbl">Cost reduction vs frontier LLM</div></div>
  <div class="kpi"><div class="val">100ms</div><div class="lbl">Avg Jev latency (vs 2,800ms)</div></div>
  <div class="kpi"><div class="val">$0</div><div class="lbl">Output token cost (parallel decode)</div></div>
  <div class="kpi"><div class="val">14ms</div><div class="lbl">CLM-8B latency (large action sets)</div></div>
</div>

<div class="insight">
<strong>Core insight:</strong> Many production AI tasks — routing, classification, scoring,
tool selection, guardrails — don't need text generation. They need fast, reliable decisions
over a known set of options. System One models are purpose-built for this: parallel output,
typed schemas, calibrated confidence, no metered output tokens.
</div>

<h3>The Three Tiers of the Emerging AI Stack</h3>
<div class="arch-box">
  <div class="arch-title">Tier 1 · Explicit Rules (code)</div>
  <p style="font-size:13px;color:var(--muted);margin:0">
  For deterministic logic: <code>if balance &lt; 0: require_payment()</code>.
  Zero AI cost. Always use when logic can be expressed as rules.
  </p>
</div>
<div class="arch-box" style="border-color:var(--accent2)">
  <div class="arch-title" style="color:var(--accent2)">Tier 2 · System One Decision Models (Jev / Laya / CLM)</div>
  <p style="font-size:13px;color:var(--muted);margin:0">
  For fuzzy choices with known option sets: routing, classification, scoring, guardrails,
  tool selection. 14–120ms, $0.000001–0.000042 per call. Parallel decode, calibrated confidence.
  </p>
</div>
<div class="arch-box" style="border-color:#8b6af7">
  <div class="arch-title" style="color:#8b6af7">Tier 3 · Generative LLMs (GPT-4o / Claude / Gemini)</div>
  <p style="font-size:13px;color:var(--muted);margin:0">
  For open-ended generation: writing code, drafting emails, complex reasoning, synthesis.
  2,000–5,000ms, $0.0003–0.005 per call. Reserve for tasks that truly need generation.
  </p>
</div>

<h3>Why System One Models Are Faster & Cheaper</h3>
<table class="cmp">
  <tr>
    <th>Property</th><th>LLM (autoregressive)</th><th>System One (Jev / Laya / CLM)</th>
  </tr>
  <tr>
    <td>Output mechanism</td>
    <td>Token by token (sequential)</td>
    <td>All outputs in parallel (one forward pass)</td>
  </tr>
  <tr>
    <td>Output type</td>
    <td>Free-form text → parse it</td>
    <td>Typed values from your schema (no parsing)</td>
  </tr>
  <tr>
    <td>Latency</td>
    <td>600–5,000ms</td>
    <td>14–120ms</td>
  </tr>
  <tr>
    <td>Cost (decision task)</td>
    <td>$0.0003–0.005 / call</td>
    <td>$0.000001–0.000042 / call</td>
  </tr>
  <tr>
    <td>Hallucination</td>
    <td>Can invent invalid answers</td>
    <td>Constrained to schema options only</td>
  </tr>
  <tr>
    <td>Confidence calibration</td>
    <td>Unreliable (logprob hacks)</td>
    <td>RLCD-trained calibrated probability</td>
  </tr>
  <tr>
    <td>Self-hosting</td>
    <td>Expensive (70B+ params)</td>
    <td>Laya: 421M params, 33ms on T4</td>
  </tr>
</table>

<div class="insight" style="margin-top:20px">
<strong>When NOT to use System One:</strong> Tasks requiring generation, synthesis, novel reasoning,
multi-step chains, or open-ended exploration still belong to Tier 3 LLMs.
If you can't enumerate the correct answers in advance, System One won't help.
</div>
</div>
"""


def benchmark_chart_html() -> str:
    max_latency = max(d["latency_ms"] for d in BENCHMARK_DATA)
    max_cost = max(d["cost_per_m_input"] + d["cost_per_m_output"] for d in BENCHMARK_DATA)

    latency_bars = ""
    cost_bars = ""
    accuracy_bars = ""

    category_colors = {
        "Frontier LLM": ("bar-llm", "🔶"),
        "Fast LLM": ("bar-llm", "🟡"),
        "System One": ("bar-s1", "🔷"),
        "System One (open)": ("bar-open", "🟢"),
        "System One (research)": ("bar-open", "🟩"),
    }

    for d in BENCHMARK_DATA:
        css, icon = category_colors.get(d["category"], ("bar-s1", "⬜"))
        lat_pct = (d["latency_ms"] / max_latency) * 100
        cost_total = d["cost_per_m_input"] + d["cost_per_m_output"]
        cost_pct = (cost_total / max_cost) * 100
        acc_pct = d["accuracy_routing"]
        label = f"{icon} {d['model']}"
        latency_bars += f"""
<div class="bar-row">
  <div class="bar-label">{label}</div>
  <div class="bar-wrap">
    <div class="bar {css}" style="width:{lat_pct:.1f}%">{d['latency_ms']}ms</div>
  </div>
</div>"""
        cost_bars += f"""
<div class="bar-row">
  <div class="bar-label">{label}</div>
  <div class="bar-wrap">
    <div class="bar {css}" style="width:{max(cost_pct,3):.1f}%">${cost_total:.3f}/M</div>
  </div>
</div>"""
        accuracy_bars += f"""
<div class="bar-row">
  <div class="bar-label">{label}</div>
  <div class="bar-wrap">
    <div class="bar {css}" style="width:{acc_pct:.1f}%">{acc_pct:.1f}%</div>
  </div>
</div>"""

    pill_legend = "".join(
        f'<span class="pill {"pill-llm" if "LLM" in c else "pill-open" if "open" in c or "research" in c else "pill-s1"}">{icon} {c}</span>'
        for c, (_, icon) in category_colors.items()
    )

    return _CSS + f"""
<div style="max-width:860px;margin:0 auto">
<h2>📊 Latency, Cost & Accuracy Benchmarks</h2>
<p style="color:var(--muted);font-size:13px">Decision task benchmarks across frontier LLMs and System One models.
Accuracy measured on support ticket routing task (n=500, mixed difficulty).</p>
<div style="margin-bottom:16px">{pill_legend}</div>

<h3>⏱ End-to-End Latency (ms) — lower is better</h3>
{latency_bars}

<h3>💰 Total API Cost per 1M Input Tokens (USD) — lower is better</h3>
{cost_bars}

<h3>🎯 Routing Accuracy (%) — higher is better</h3>
{accuracy_bars}

<div class="insight" style="margin-top:20px">
<strong>Key finding:</strong> Jev delivers 93.8% accuracy (vs GPT-4o's 94.2%) at 40× lower latency
and 444× lower cost. Laya (open-source, self-hosted) trades 4 accuracy points for zero API cost and 33ms latency.
CLM-8B's contrastive approach excels when the action set is large (100+ options).
</div>
<table class="cmp" style="margin-top:16px">
  <tr><th>Model</th><th>Latency</th><th>Input Cost /M</th><th>Output Cost /M</th><th>Accuracy</th><th>Notes</th></tr>
  {"".join(f'<tr><td><strong>{d["model"]}</strong></td><td>{d["latency_ms"]}ms</td><td>${d["cost_per_m_input"]:.3f}</td><td>${d["cost_per_m_output"]:.2f}</td><td>{d["accuracy_routing"]}%</td><td style="color:var(--muted);font-size:12px">{d["note"]}</td></tr>' for d in BENCHMARK_DATA)}
</table>
</div>
"""


def schema_html(use_case_name: str) -> str:
    schema = next((s for s in DECISION_SCHEMAS if s["use_case"] == use_case_name), DECISION_SCHEMAS[0])
    fields_html = ""
    for key, values in schema["schema"].items():
        vals_html = "".join(f'<span class="tag">{v}</span>' for v in values)
        fields_html += f"""
<div class="schema-field">
  <span class="schema-key">{key}</span>
  <span class="schema-values">{vals_html}</span>
</div>"""
    savings_factor = schema["llm_equivalent_tokens"] / max(schema["system_one_tokens"], 1)
    return _CSS + f"""
<div style="max-width:860px;margin:0 auto">
<h2>📋 Decision Schema: {schema["use_case"]}</h2>
<p style="color:var(--muted);font-size:13px">{schema["input_description"]}</p>

<div class="card">
<div style="margin-bottom:12px;font-size:13px;color:var(--muted)">
You define the schema. The model returns typed values — no parsing, no hallucination outside schema.
</div>
{fields_html}
</div>

<h3>📝 Example Input</h3>
<div class="arch-box">
  <p style="font-size:13px;font-style:italic;margin:0">"{schema["example_input"]}"</p>
</div>

<h3>✅ Expected Typed Output</h3>
<div class="result-box">
{"".join(f'<div class="result-row"><span class="result-key">{k}</span><span class="result-val">{v}</span></div>' for k, v in schema["expected_output"].items())}
</div>

<div class="kpi-row" style="margin-top:16px">
  <div class="kpi">
    <div class="val">{schema["llm_equivalent_tokens"]}</div>
    <div class="lbl">Tokens with LLM (prompt + output)</div>
  </div>
  <div class="kpi">
    <div class="val">{schema["system_one_tokens"]}</div>
    <div class="lbl">Tokens with System One</div>
  </div>
  <div class="kpi">
    <div class="val">{savings_factor:.0f}×</div>
    <div class="lbl">Token reduction</div>
  </div>
</div>
</div>
"""


def routing_demo_html(example_index: int = 0) -> str:
    ex = ROUTING_EXAMPLES[min(example_index, len(ROUTING_EXAMPLES) - 1)]
    result_rows = "".join(
        f'<div class="result-row"><span class="result-key">{k}</span><span class="result-val">{v}</span></div>'
        for k, v in ex["result"].items()
    )
    conf_pct = int(ex["confidence"] * 100)
    cost_reduction = int(ex["cost_llm_usd"] / max(ex["cost_jev_usd"], 1e-10))
    latency_reduction = int(ex["latency_llm_ms"] / max(ex["latency_jev_ms"], 1))
    return _CSS + f"""
<div style="max-width:860px;margin:0 auto">
<h2>⚡ Live Routing Demo</h2>
<p style="color:var(--muted);font-size:13px">Select an example above. Simulated Jev System One decision (no live API call).</p>

<div class="card">
  <div style="font-size:12px;color:var(--muted);margin-bottom:6px">INPUT TEXT · Schema: {ex["schema"]}</div>
  <p style="font-style:italic;margin:0">"{ex["text"]}"</p>
</div>

<h3>🎯 Decision Output</h3>
<div class="result-box">
{result_rows}
</div>

<h3>📊 Confidence: {conf_pct}%</h3>
<div class="conf-bar"><div class="conf-fill" style="width:{conf_pct}%"></div></div>
<p style="font-size:12px;color:var(--muted);margin-top:4px">
RLCD-calibrated: {conf_pct}% confidence = {conf_pct}% correct in Jev's training distribution.
{"Auto-act threshold exceeded — no human review needed." if ex["confidence"] >= 0.95 else "Below 95% threshold — queued for review."}
</p>

<div class="kpi-row">
  <div class="kpi">
    <div class="val">{ex["latency_jev_ms"]}ms</div>
    <div class="lbl">Jev latency</div>
  </div>
  <div class="kpi">
    <div class="val">{ex["latency_llm_ms"]}ms</div>
    <div class="lbl">LLM latency</div>
  </div>
  <div class="kpi">
    <div class="val">{latency_reduction}×</div>
    <div class="lbl">Latency reduction</div>
  </div>
  <div class="kpi">
    <div class="val">{cost_reduction:,}×</div>
    <div class="lbl">Cost reduction</div>
  </div>
</div>

<div class="insight">
At 1M decisions/day: Jev costs <strong>${ex["cost_jev_usd"] * 1_000_000:.2f}/day</strong> vs LLM
<strong>${ex["cost_llm_usd"] * 1_000_000:.2f}/day</strong> — a saving of
<strong>${(ex["cost_llm_usd"] - ex["cost_jev_usd"]) * 1_000_000 * 365:.0f}/year</strong> at this traffic level.
</div>
</div>
"""


def example_labels() -> list[str]:
    return [f"{i+1}. [{ex['schema']}] {ex['text'][:60]}…" for i, ex in enumerate(ROUTING_EXAMPLES)]
