"""
Jev Memory Compaction: Scoring and Dropping Irrelevant Tool Calls
Source: TypesafeAI's Jev · AlphaSignal Signal · 2026-09-18
Score: 8/10 — Drop-in context window management for agent pipelines
"""

import math

# ── Context bloat data ────────────────────────────────────────────────────────

CONTEXT_BLOAT_DATA = [
    {"session_steps": 5,  "raw_tokens": 2_400,  "compacted_tokens": 2_100,  "savings_pct": 12.5},
    {"session_steps": 10, "raw_tokens": 6_800,  "compacted_tokens": 4_200,  "savings_pct": 38.2},
    {"session_steps": 20, "raw_tokens": 18_500, "compacted_tokens": 7_800,  "savings_pct": 57.8},
    {"session_steps": 40, "raw_tokens": 48_000, "compacted_tokens": 11_200, "savings_pct": 76.7},
    {"session_steps": 80, "raw_tokens": 112_000,"compacted_tokens": 14_900, "savings_pct": 86.7},
]

# ── Scoring weights ───────────────────────────────────────────────────────────

SCORING_WEIGHTS = {
    "recency":       {"weight": 0.40, "description": "How recent is the call? Older calls decay exponentially."},
    "importance":    {"weight": 0.35, "description": "How many subsequent calls referenced this call's output?"},
    "task_alignment":{"weight": 0.25, "description": "Cosine similarity of call embedding vs current task embedding."},
}

# ── Example tool calls for the live demo ─────────────────────────────────────

TOOL_CALL_EXAMPLES = [
    {
        "id": 1, "step": 1,
        "tool": "web_search",
        "input": "latest RAG techniques 2026",
        "output_tokens": 420,
        "referenced_by": [3, 5, 7],
        "task_keywords": ["RAG", "retrieval", "search"],
    },
    {
        "id": 2, "step": 2,
        "tool": "read_file",
        "input": "README.md",
        "output_tokens": 180,
        "referenced_by": [],
        "task_keywords": [],
    },
    {
        "id": 3, "step": 3,
        "tool": "web_search",
        "input": "hybrid search BM25 dense retrieval fusion",
        "output_tokens": 530,
        "referenced_by": [5, 6, 8],
        "task_keywords": ["hybrid", "search", "BM25", "dense"],
    },
    {
        "id": 4, "step": 4,
        "tool": "bash",
        "input": "ls /tmp",
        "output_tokens": 60,
        "referenced_by": [],
        "task_keywords": [],
    },
    {
        "id": 5, "step": 5,
        "tool": "web_search",
        "input": "alpha parameter tuning hybrid search",
        "output_tokens": 390,
        "referenced_by": [7, 8],
        "task_keywords": ["alpha", "tuning", "hybrid", "search"],
    },
    {
        "id": 6, "step": 6,
        "tool": "read_file",
        "input": "notes_draft.txt",
        "output_tokens": 240,
        "referenced_by": [8],
        "task_keywords": ["hybrid", "search"],
    },
    {
        "id": 7, "step": 7,
        "tool": "web_search",
        "input": "RRF reciprocal rank fusion implementation",
        "output_tokens": 470,
        "referenced_by": [8],
        "task_keywords": ["RRF", "fusion", "retrieval"],
    },
    {
        "id": 8, "step": 8,
        "tool": "write_file",
        "input": "hybrid_search_report.md",
        "output_tokens": 880,
        "referenced_by": [],
        "task_keywords": ["hybrid", "search", "RAG", "retrieval"],
    },
]

# ── Scoring logic ─────────────────────────────────────────────────────────────

def _recency_score(step: int, total_steps: int, half_life: float = 4.0) -> float:
    age = total_steps - step
    return math.exp(-age * math.log(2) / half_life)


def _importance_score(referenced_by: list, total_steps: int) -> float:
    if total_steps <= 1:
        return 0.0
    return min(1.0, len(referenced_by) / max(1, total_steps - 1) * 3)


def _task_alignment_score(call_keywords: list, task_query: str) -> float:
    if not call_keywords or not task_query:
        return 0.0
    task_tokens = set(task_query.lower().split())
    call_tokens = set(k.lower() for k in call_keywords)
    overlap = len(call_tokens & task_tokens)
    union = len(call_tokens | task_tokens)
    return overlap / union if union > 0 else 0.0


def score_tool_calls(
    calls: list,
    task_query: str = "hybrid search RAG retrieval implementation",
    w_recency: float = 0.40,
    w_importance: float = 0.35,
    w_alignment: float = 0.25,
) -> list:
    total = len(calls)
    scored = []
    for c in calls:
        r = _recency_score(c["step"], total)
        i = _importance_score(c.get("referenced_by", []), total)
        a = _task_alignment_score(c.get("task_keywords", []), task_query)
        score = w_recency * r + w_importance * i + w_alignment * a
        scored.append({**c, "score_recency": r, "score_importance": i,
                       "score_alignment": a, "composite_score": score})
    return sorted(scored, key=lambda x: x["composite_score"], reverse=True)


def compact_context(calls: list, threshold: float = 0.30, task_query: str = "") -> dict:
    scored = score_tool_calls(calls, task_query=task_query)
    kept   = [c for c in scored if c["composite_score"] >= threshold]
    dropped = [c for c in scored if c["composite_score"] < threshold]
    raw_tokens  = sum(c["output_tokens"] for c in calls)
    kept_tokens = sum(c["output_tokens"] for c in kept)
    return {
        "kept": sorted(kept,    key=lambda x: x["step"]),
        "dropped": sorted(dropped, key=lambda x: x["step"]),
        "raw_tokens": raw_tokens,
        "compacted_tokens": kept_tokens,
        "savings_pct": round((1 - kept_tokens / max(raw_tokens, 1)) * 100, 1),
    }

# ── HTML chart helpers ────────────────────────────────────────────────────────

def bloat_chart_html() -> str:
    bars = ""
    max_raw = max(d["raw_tokens"] for d in CONTEXT_BLOAT_DATA)
    for d in CONTEXT_BLOAT_DATA:
        raw_w  = int(d["raw_tokens"] / max_raw * 380)
        comp_w = int(d["compacted_tokens"] / max_raw * 380)
        bars += f"""
        <tr>
          <td style='padding:6px 10px;color:#9ca3af;width:80px'>{d['session_steps']} steps</td>
          <td style='padding:6px 4px'>
            <div style='background:#ef4444;height:18px;width:{raw_w}px;border-radius:3px;display:inline-block'></div>
            <span style='color:#ef4444;margin-left:6px;font-size:12px'>{d['raw_tokens']:,}</span>
          </td>
          <td style='padding:6px 4px'>
            <div style='background:#22c55e;height:18px;width:{comp_w}px;border-radius:3px;display:inline-block'></div>
            <span style='color:#22c55e;margin-left:6px;font-size:12px'>{d['compacted_tokens']:,}</span>
          </td>
          <td style='padding:6px 10px;color:#f59e0b;font-weight:bold;font-size:13px'>−{d['savings_pct']}%</td>
        </tr>"""
    return f"""
    <div style='background:#111827;border-radius:10px;padding:20px;font-family:monospace'>
      <div style='display:flex;gap:20px;margin-bottom:14px'>
        <span style='color:#ef4444'>■ Raw context</span>
        <span style='color:#22c55e'>■ After Jev compaction</span>
        <span style='color:#f59e0b'>● Token savings</span>
      </div>
      <table style='border-collapse:collapse;width:100%'>{bars}</table>
      <p style='color:#6b7280;font-size:12px;margin-top:12px'>
        Token counts grow super-linearly without compaction — a 80-step session balloons to
        112K tokens. Jev holds it under 15K by continuously scoring and evicting low-relevance calls.
      </p>
    </div>"""


def scoring_html(scored_calls: list | None = None) -> str:
    if scored_calls is None:
        scored_calls = score_tool_calls(TOOL_CALL_EXAMPLES)
    rows = ""
    for c in sorted(scored_calls, key=lambda x: x["step"]):
        s = c["composite_score"]
        bar_w = int(s * 180)
        color = "#22c55e" if s >= 0.5 else "#f59e0b" if s >= 0.3 else "#ef4444"
        keep = "KEEP" if s >= 0.30 else "DROP"
        keep_color = "#22c55e" if keep == "KEEP" else "#ef4444"
        rows += f"""
        <tr style='border-bottom:1px solid #1f2937'>
          <td style='padding:8px 10px;color:#9ca3af'>#{c['step']}</td>
          <td style='padding:8px 10px;color:#60a5fa'>{c['tool']}</td>
          <td style='padding:8px 10px;color:#d1d5db;max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'
              title='{c["input"]}'>{c['input'][:40]}…</td>
          <td style='padding:8px 10px;color:#9ca3af'>{c['output_tokens']}</td>
          <td style='padding:8px 10px'>
            <div style='background:{color};height:14px;width:{bar_w}px;border-radius:3px;display:inline-block'></div>
            <span style='color:{color};margin-left:6px;font-size:12px'>{s:.2f}</span>
          </td>
          <td style='padding:8px 10px;font-weight:bold;color:{keep_color}'>{keep}</td>
        </tr>"""
    return f"""
    <div style='background:#111827;border-radius:10px;padding:20px;font-family:monospace;overflow-x:auto'>
      <table style='border-collapse:collapse;width:100%;min-width:620px'>
        <thead>
          <tr style='border-bottom:2px solid #374151'>
            <th style='padding:8px 10px;color:#6b7280;text-align:left'>Step</th>
            <th style='padding:8px 10px;color:#6b7280;text-align:left'>Tool</th>
            <th style='padding:8px 10px;color:#6b7280;text-align:left'>Input</th>
            <th style='padding:8px 10px;color:#6b7280;text-align:left'>Tokens</th>
            <th style='padding:8px 10px;color:#6b7280;text-align:left'>Score (0–1)</th>
            <th style='padding:8px 10px;color:#6b7280;text-align:left'>Decision</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


# ── Implementation patterns ───────────────────────────────────────────────────

IMPL_PATTERNS = [
    {
        "pattern": "Pattern 1 — Passive Logging (Zero overhead)",
        "description": (
            "Record every tool call's metadata (step, tool name, token count, referenced calls) "
            "to a lightweight sidecar log. No compaction yet — just instrumentation. "
            "This is the prerequisite for all subsequent patterns and costs nothing at runtime."
        ),
        "when": "Greenfield agent where you control the tool-calling loop.",
        "code": """\
import json, time
from pathlib import Path

class JevLogger:
    def __init__(self, session_id: str):
        self.log = []
        self.step = 0
        self.path = Path(f"jev_{session_id}.jsonl")

    def record(self, tool: str, input_str: str, output: str, refs: list[int] = None):
        self.step += 1
        entry = {
            "step": self.step, "tool": tool,
            "input": input_str[:200], "output_tokens": len(output.split()),
            "referenced_by": refs or [], "ts": time.time(),
        }
        self.log.append(entry)
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\\n")
        return entry""",
    },
    {
        "pattern": "Pattern 2 — Threshold Compaction (Drop on score < θ)",
        "description": (
            "After every K tool calls, run the Jev scorer over the accumulated history. "
            "Remove any call whose composite_score < θ (default 0.30) from the context "
            "window passed to the next LLM call. Dropped calls are archived, not deleted — "
            "they can be fetched back if a future call references them."
        ),
        "when": "Any agent pipeline where you assemble the context window before each LLM call.",
        "code": """\
import math

def jev_score(call: dict, all_calls: list, task_query: str, total: int) -> float:
    age = total - call["step"]
    recency    = math.exp(-age * math.log(2) / 4.0)
    importance = min(1.0, len(call.get("referenced_by", [])) / max(1, total - 1) * 3)
    task_tok   = set(task_query.lower().split())
    call_tok   = set(k.lower() for k in call.get("task_keywords", []))
    alignment  = len(call_tok & task_tok) / len(call_tok | task_tok) if call_tok | task_tok else 0
    return 0.40 * recency + 0.35 * importance + 0.25 * alignment

def compact(calls: list, task_query: str, threshold: float = 0.30) -> list:
    total = len(calls)
    return [c for c in calls if jev_score(c, calls, task_query, total) >= threshold]""",
    },
    {
        "pattern": "Pattern 3 — Budget-Capped Compaction (Keep top-K by token budget)",
        "description": (
            "Instead of a fixed score threshold, sort calls by score descending and "
            "greedily select them until a token budget is reached. "
            "This is safer than threshold compaction when you need a hard context limit "
            "(e.g., always stay under 8K tokens regardless of session length)."
        ),
        "when": "Production agents with strict context-window cost budgets.",
        "code": """\
def compact_budget(scored_calls: list, token_budget: int = 8_000) -> list:
    sorted_calls = sorted(scored_calls, key=lambda c: c["composite_score"], reverse=True)
    kept, used = [], 0
    for c in sorted_calls:
        tok = c.get("output_tokens", 100)
        if used + tok <= token_budget:
            kept.append(c)
            used += tok
    return sorted(kept, key=lambda c: c["step"])""",
    },
    {
        "pattern": "Pattern 4 — Black-Box Wrapper (Intercept context before any LLM call)",
        "description": (
            "Wrap an existing agent's LLM client. Before every call, intercept the messages "
            "list, extract tool-result messages, score them, and drop low-scorers. "
            "The underlying agent never knows compaction happened — pure middleware."
        ),
        "when": "Wrapping a third-party agent (LangChain, LlamaIndex, CrewAI) without modifying its internals.",
        "code": """\
class JevMiddleware:
    def __init__(self, llm_client, threshold: float = 0.30):
        self.client = llm_client
        self.threshold = threshold
        self._step = 0

    def chat(self, messages: list, **kw):
        compacted = self._compact(messages)
        return self.client.chat(compacted, **kw)

    def _compact(self, messages: list) -> list:
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        other     = [m for m in messages if m.get("role") != "tool"]
        if len(tool_msgs) < 4:
            return messages  # nothing to compact yet
        scored = _score_messages(tool_msgs, total=len(tool_msgs))
        kept = [m for m, s in scored if s >= self.threshold]
        return other + kept""",
    },
]
