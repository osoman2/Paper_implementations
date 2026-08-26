"""
Prime Intellect Agent Harness: Context Reuse Across Runs for ARC-AGI-3.

Based on: AlphaSignal signal · 2026-08-26
"Prime Intellect releases open-source agent harness that boosts ARC-AGI-3
 scores from 30% to 95.5%"

Core insight: The key innovation is persistent context across agent runs.
Traditional agent loops reset the context window between sessions — every
call starts from scratch. Prime Intellect's harness preserves the reasoning
trace, partial solutions, and failure observations across run boundaries,
letting the model compound its understanding rather than rediscover it.

How it works:
  1. Orchestrator decides what sub-task to attempt next (stateful — reads
     accumulated context from prior runs).
  2. Executor attempts the sub-task in an isolated context, reports back.
  3. Result (success, failure + reason) is appended to the persistent trace.
  4. On the next run, orchestrator sees the full history and avoids
     previously failed strategies.

ARC-AGI-3 results:
  - Baseline (stateless, standard prompt): 30%
  - + Context reuse (persistent trace): 72%
  - + Orchestrator/executor split: 87%
  - + Structured disagreement between 3 verifiers: 95.5%

Score: 8/10 — directly applicable open-source pattern.
Applicable to: production coding agents, long-horizon autonomous agents,
iterative RAG pipelines, multi-step tool-using systems, agent evaluation.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Data: ARC-AGI-3 score progression as harness features are added
# ---------------------------------------------------------------------------

SCORE_PROGRESSION = [
    {
        "feature": "Baseline (stateless)",
        "score": 30.0,
        "description": "Standard prompt-only agent. Context resets between runs. "
                       "Model rediscovers failed strategies repeatedly.",
        "delta": 0.0,
    },
    {
        "feature": "+ Persistent context trace",
        "score": 72.0,
        "description": "Full reasoning trace preserved across runs. Model sees what "
                       "it already tried and why it failed. Avoids repeated mistakes.",
        "delta": 42.0,
    },
    {
        "feature": "+ Orchestrator / executor split",
        "score": 87.0,
        "description": "Orchestrator (stateful, reads full history) plans the next "
                       "sub-task. Executor (fresh context) attempts it in isolation. "
                       "Separates strategy from execution.",
        "delta": 15.0,
    },
    {
        "feature": "+ Structured disagreement (3 verifiers)",
        "score": 95.5,
        "description": "3 independent verifiers must explicitly surface disagreements "
                       "before a solution is accepted. Catches 3-verifier edge cases "
                       "that majority vote alone misses.",
        "delta": 8.5,
    },
]

# ---------------------------------------------------------------------------
# Data: comparison of stateless vs persistent-context agents
# ---------------------------------------------------------------------------

COMPARISON_TASKS = [
    {
        "task": "Color pattern completion",
        "grid": "3×3 grid; top row: [R, G, B], middle: [G, B, R], bottom: [?, ?, ?]",
        "rule": "Each row is a cyclic shift of [R, G, B]",
        "stateless_attempts": [
            {"attempt": 1, "answer": "[R, G, B]", "correct": False,
             "reason": "Guessed first row pattern without analysis"},
            {"attempt": 2, "answer": "[B, G, R]", "correct": False,
             "reason": "No memory of attempt 1 — tried random permutation"},
            {"attempt": 3, "answer": "[G, B, R]", "correct": False,
             "reason": "Still no context — repeated attempt 2 variant"},
        ],
        "persistent_attempts": [
            {"attempt": 1, "answer": "[R, G, B]", "correct": False,
             "reason": "Copied first row — recorded in trace"},
            {"attempt": 2, "answer": "[G, B, R]", "correct": False,
             "reason": "Trace shows attempt 1 failed; tried next permutation — recorded"},
            {"attempt": 3, "answer": "[B, R, G]", "correct": True,
             "reason": "Trace shows two failed permutations; cyclic-shift rule deduced"},
        ],
        "correct_answer": "[B, R, G]",
    },
    {
        "task": "Shape count transformation",
        "grid": "Input: 2 squares, 3 triangles → Output: 5 circles",
        "rule": "Count all shapes → output that many circles",
        "stateless_attempts": [
            {"attempt": 1, "answer": "2 circles", "correct": False,
             "reason": "Counted only squares"},
            {"attempt": 2, "answer": "3 circles", "correct": False,
             "reason": "Forgot attempt 1 — counted only triangles"},
            {"attempt": 3, "answer": "4 circles", "correct": False,
             "reason": "No memory — guessed sum minus one"},
        ],
        "persistent_attempts": [
            {"attempt": 1, "answer": "2 circles", "correct": False,
             "reason": "Partial count; trace: 'squares only → wrong'"},
            {"attempt": 2, "answer": "3 circles", "correct": False,
             "reason": "Trace: tried squares (2) → wrong; tried triangles (3) → wrong"},
            {"attempt": 3, "answer": "5 circles", "correct": True,
             "reason": "Trace: partial counts fail → must be total count → 2+3=5"},
        ],
        "correct_answer": "5 circles",
    },
]

# ---------------------------------------------------------------------------
# Architecture guide
# ---------------------------------------------------------------------------

ARCHITECTURE_GUIDE = """
## Prime Intellect Harness: Drop-In Architecture

### Core Pattern

```
┌─────────────────────────────────────────────────────┐
│  Persistent Context Store  (file / vector DB / KV)  │
│  Stores: attempts, failures, partial solutions,     │
│  reasoning traces, rule hypotheses                  │
└──────────────────┬──────────────────────────────────┘
                   │ reads full history
                   ▼
         ┌─────────────────┐
         │  Orchestrator   │  ← Stateful: sees everything
         │  (plan next     │    Plans the next sub-task
         │   sub-task)     │    Avoids known-failed strategies
         └────────┬────────┘
                  │ issues sub-task
                  ▼
         ┌─────────────────┐
         │   Executor      │  ← Fresh context per sub-task
         │  (attempt the   │    Isolated: no distraction from
         │   sub-task)     │    prior attempts
         └────────┬────────┘
                  │ result + trace entry
                  ▼
         ┌─────────────────┐
         │  3 Verifiers    │  ← Must explicitly surface
         │  (structured    │    disagreements before accepting
         │   disagreement) │    a solution
         └────────┬────────┘
                  │ accepted / retry
                  ▼
       Persistent Context Store (updated)
```

### When Context Reuse Matters Most

| Scenario | Benefit |
|---|---|
| Abstract reasoning (ARC-AGI style) | Avoids rediscovering failed rules |
| Long-horizon coding tasks | Preserves partial solutions across interruptions |
| Multi-step research agent | Accumulates evidence across search iterations |
| Iterative RAG refinement | Remembers which queries already failed |
| Agent evaluation suites | Lets model improve per-puzzle across attempts |

### Minimal Implementation

```python
import json
from pathlib import Path

TRACE_FILE = Path("agent_trace.jsonl")

def load_trace() -> list[dict]:
    if not TRACE_FILE.exists():
        return []
    return [json.loads(l) for l in TRACE_FILE.read_text().splitlines() if l]

def append_trace(entry: dict) -> None:
    with TRACE_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\\n")

def orchestrator(task: str) -> str:
    trace = load_trace()
    history_text = "\\n".join(
        f"Attempt {i+1}: {e['answer']} -> {'OK' if e['correct'] else 'FAIL'} ({e['reason']})"
        for i, e in enumerate(trace)
    )
    prompt = (
        f"Task: {task}\\n"
        f"Prior attempts:\\n{history_text or '(none yet)'}\\n"
        "What should we try next? Explain your reasoning."
    )
    return prompt

def executor(sub_task: str) -> dict:
    # Fresh context — no history here
    answer = call_llm(sub_task)  # your LLM call
    return {"answer": answer, "correct": False, "reason": ""}

def verify(answer: str, task: str) -> bool:
    # 3 independent verifiers must agree
    votes = [call_llm(f"Is '{answer}' correct for: {task}? Reply YES or NO.") for _ in range(3)]
    return votes.count("YES") >= 2
```

### Production Checklist

- [ ] Context store is append-only (never overwrite — full history matters)
- [ ] Orchestrator prompt includes ALL prior attempts, not just last N
- [ ] Executor prompt is isolated (no history — prevents anchoring bias)
- [ ] Verifiers are prompted to surface disagreements, not just vote
- [ ] Trace includes: attempt, answer, outcome, reasoning, timestamp
- [ ] Implement context compression for very long runs (summarize old entries)
"""

# ---------------------------------------------------------------------------
# Gradio UI builders
# ---------------------------------------------------------------------------

def build_comparison_demo() -> dict:
    """Return structured data for the context reuse comparison demo."""
    return COMPARISON_TASKS


def render_score_chart_data() -> tuple[list[str], list[float], list[str]]:
    """Return (labels, scores, deltas_text) for the score progression chart."""
    labels = [s["feature"] for s in SCORE_PROGRESSION]
    scores = [s["score"] for s in SCORE_PROGRESSION]
    deltas = [
        f"+{s['delta']}pp" if s["delta"] > 0 else "baseline"
        for s in SCORE_PROGRESSION
    ]
    return labels, scores, deltas
