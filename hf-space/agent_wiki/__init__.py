"""
Agent Wiki: Persistent Knowledge Wiki for Small Agents

Based on: AlphaSignal signal · 2026-09-03 PM
"Google's framework lets small AI agents outperform models 3x their size using
a persistent knowledge wiki"
  - 406 AlphaSignal signal likes
  - Novel architectural framework: small agents + persistent write-capable wiki
    outperform models 3x their parameter count on multi-step reasoning tasks
  - Wiki acts as external long-term memory: agents read/write facts, plans, and
    observations across turns — analogous to RAG but write-capable + agent-controlled

Score: 9/10 — novel, directly implementable, high architectural impact for any
agentic RAG system where long-horizon memory is the bottleneck.

Key finding: Architecture beats scale — a 7B agent with a persistent wiki
outperforms a 21B model on multi-step tasks because the wiki offloads memory
to structured external storage, freeing context for reasoning.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Wiki entry model
# ---------------------------------------------------------------------------

@dataclass
class WikiEntry:
    key: str
    value: str
    kind: str          # "fact", "plan", "observation", "conclusion"
    turn: int
    confidence: float  # 0.0–1.0
    source: str        # which agent wrote this

# ---------------------------------------------------------------------------
# Simulated task scenarios
# ---------------------------------------------------------------------------

TASKS: list[dict] = [
    {
        "id": "debug_latency",
        "label": "Debug API latency spike (multi-step)",
        "description": (
            "An API endpoint suddenly shows P99 latency > 2 s. Diagnose root cause "
            "and propose a fix. Requires correlating metrics, logs, and code changes "
            "across 5 turns."
        ),
        "turns": [
            {
                "turn": 1,
                "action": "Read metrics dashboard",
                "wiki_writes": [
                    WikiEntry("symptom.p99_latency", "P99 = 2.3 s (baseline 120 ms)", "fact", 1, 0.99, "agent_7b"),
                    WikiEntry("symptom.start_time",  "Spike began 14:22 UTC",          "fact", 1, 0.99, "agent_7b"),
                    WikiEntry("plan.step1", "Check recent deployments near 14:22 UTC", "plan", 1, 0.90, "agent_7b"),
                ],
                "agent_7b_answer": (
                    "Recorded: P99 = 2.3 s (normal: 120 ms), spike at 14:22 UTC. "
                    "Plan written to wiki: check deployments."
                ),
                "agent_21b_answer": (
                    "P99 latency is 2.3 s. There was likely a traffic spike or a "
                    "slow database query. I'll check the logs."
                ),
            },
            {
                "turn": 2,
                "action": "Check recent deployments",
                "wiki_reads": ["plan.step1", "symptom.start_time"],
                "wiki_writes": [
                    WikiEntry("deploy.recent", "v2.4.1 deployed 14:19 UTC — added synchronous external API call to /checkout", "fact", 2, 0.97, "agent_7b"),
                    WikiEntry("plan.step2",    "Profile /checkout endpoint for blocking I/O", "plan", 2, 0.92, "agent_7b"),
                ],
                "agent_7b_answer": (
                    "Wiki says spike at 14:22 UTC → deployment at 14:19 UTC "
                    "(v2.4.1) added a synchronous external API call to /checkout. "
                    "Next: profile /checkout for blocking I/O."
                ),
                "agent_21b_answer": (
                    "Checking deployments... v2.4.1 was deployed recently. "
                    "It might be related. Let me look at the logs again."
                ),
            },
            {
                "turn": 3,
                "action": "Profile /checkout endpoint",
                "wiki_reads": ["deploy.recent", "plan.step2"],
                "wiki_writes": [
                    WikiEntry("root_cause.io", "PaymentGateway.verify() blocks for 1.9 s avg — no timeout set", "fact", 3, 0.98, "agent_7b"),
                    WikiEntry("plan.step3",    "Fix: add 500 ms timeout + async wrapper to PaymentGateway.verify()", "plan", 3, 0.95, "agent_7b"),
                ],
                "agent_7b_answer": (
                    "Wiki: /checkout has new external call. Profiling shows "
                    "PaymentGateway.verify() blocks avg 1.9 s — no timeout. "
                    "Root cause confirmed. Fix written to wiki."
                ),
                "agent_21b_answer": (
                    "Profiling shows slow execution. There might be a blocking "
                    "call somewhere. Hard to tell without seeing the full context."
                ),
            },
            {
                "turn": 4,
                "action": "Implement fix",
                "wiki_reads": ["root_cause.io", "plan.step3"],
                "wiki_writes": [
                    WikiEntry("fix.code", "Added asyncio.wait_for(PaymentGateway.verify(), timeout=0.5) in checkout.py:L142", "fact", 4, 0.96, "agent_7b"),
                    WikiEntry("fix.rollback", "Rollback plan: revert v2.4.1 if P99 > 500 ms after deploy", "plan", 4, 0.93, "agent_7b"),
                ],
                "agent_7b_answer": (
                    "Applied fix per wiki plan: asyncio.wait_for with 500 ms timeout "
                    "at checkout.py:L142. Rollback plan recorded in wiki."
                ),
                "agent_21b_answer": (
                    "I believe we should add a timeout. Let me write some code... "
                    "actually, I need to recall what file the call was in — "
                    "was it checkout.py?"
                ),
            },
            {
                "turn": 5,
                "action": "Validate & summarise",
                "wiki_reads": ["symptom.p99_latency", "fix.code", "fix.rollback", "root_cause.io"],
                "wiki_writes": [
                    WikiEntry("outcome.p99", "Post-fix P99 = 98 ms (−96%)", "conclusion", 5, 0.99, "agent_7b"),
                    WikiEntry("outcome.lesson", "Always set I/O timeouts on external calls in hot paths", "conclusion", 5, 0.95, "agent_7b"),
                ],
                "agent_7b_answer": (
                    "Wiki summary: root cause = blocking PaymentGateway.verify() "
                    "(1.9 s, no timeout). Fix: 500 ms async timeout → P99 dropped "
                    "from 2.3 s to 98 ms (−96%). Lesson logged."
                ),
                "agent_21b_answer": (
                    "I think the fix worked. The latency should be better now. "
                    "We might want to monitor it."
                ),
            },
        ],
    },
    {
        "id": "design_schema",
        "label": "Design a multi-tenant DB schema (multi-step)",
        "description": (
            "Design a PostgreSQL schema for a multi-tenant SaaS app: users, orgs, "
            "workspaces, and audit logs — with row-level security. 5 planning turns."
        ),
        "turns": [
            {
                "turn": 1,
                "action": "Gather requirements",
                "wiki_writes": [
                    WikiEntry("req.tenancy",  "Row-level isolation per org_id — no cross-tenant data leaks", "fact", 1, 0.99, "agent_7b"),
                    WikiEntry("req.entities", "Core entities: users, orgs, workspaces, audit_logs", "fact", 1, 0.99, "agent_7b"),
                    WikiEntry("plan.step1",   "Define org & user tables first, then workspace, then audit_logs", "plan", 1, 0.91, "agent_7b"),
                ],
                "agent_7b_answer": "Requirements captured in wiki: RLS per org_id, 4 core entities. Plan set.",
                "agent_21b_answer": "We need tables for users, orgs, and workspaces. Let me think about the schema.",
            },
            {
                "turn": 2,
                "action": "Design core tables",
                "wiki_reads": ["req.entities", "plan.step1"],
                "wiki_writes": [
                    WikiEntry("schema.orgs",  "orgs(id UUID PK, name TEXT, plan TEXT, created_at TIMESTAMPTZ)", "fact", 2, 0.97, "agent_7b"),
                    WikiEntry("schema.users", "users(id UUID PK, org_id UUID FK→orgs.id, email TEXT UNIQUE, role TEXT)", "fact", 2, 0.97, "agent_7b"),
                ],
                "agent_7b_answer": "Wiki guided design: orgs + users with org_id FK. RLS candidate column: org_id.",
                "agent_21b_answer": "Here's a users table... wait, do we need org_id in users? I need to think about this.",
            },
            {
                "turn": 3,
                "action": "Design workspaces & RLS",
                "wiki_reads": ["schema.orgs", "schema.users", "req.tenancy"],
                "wiki_writes": [
                    WikiEntry("schema.workspaces", "workspaces(id UUID PK, org_id UUID FK, name TEXT, settings JSONB)", "fact", 3, 0.96, "agent_7b"),
                    WikiEntry("schema.rls",         "ENABLE ROW LEVEL SECURITY on users, workspaces; POLICY: WHERE org_id = current_setting('app.org_id')", "fact", 3, 0.95, "agent_7b"),
                ],
                "agent_7b_answer": "workspaces table + RLS policies using app.org_id session variable. Recorded in wiki.",
                "agent_21b_answer": "Workspaces should reference orgs. For RLS... I think we use a policy but I'm unsure of exact syntax.",
            },
            {
                "turn": 4,
                "action": "Design audit_logs",
                "wiki_reads": ["schema.orgs", "schema.users"],
                "wiki_writes": [
                    WikiEntry("schema.audit_logs", "audit_logs(id BIGSERIAL PK, org_id UUID, actor_id UUID, action TEXT, payload JSONB, ts TIMESTAMPTZ DEFAULT now())", "fact", 4, 0.97, "agent_7b"),
                    WikiEntry("schema.indexes",    "Indexes: (org_id, ts DESC) on audit_logs, (org_id) on workspaces", "fact", 4, 0.94, "agent_7b"),
                ],
                "agent_7b_answer": "audit_logs designed with org_id partition key. Indexes added to wiki.",
                "agent_21b_answer": "audit_logs table with timestamps. We should probably index it somehow.",
            },
            {
                "turn": 5,
                "action": "Final review & migration plan",
                "wiki_reads": ["schema.orgs", "schema.users", "schema.workspaces", "schema.rls", "schema.audit_logs", "schema.indexes"],
                "wiki_writes": [
                    WikiEntry("outcome.schema", "4-table multi-tenant schema complete with RLS and indexes", "conclusion", 5, 0.99, "agent_7b"),
                    WikiEntry("outcome.migration", "Migration order: orgs → users → workspaces → audit_logs → RLS policies → indexes", "conclusion", 5, 0.97, "agent_7b"),
                ],
                "agent_7b_answer": "Full schema assembled from wiki. Migration order set. Schema is consistent and RLS-complete.",
                "agent_21b_answer": "The schema looks reasonable. I hope I covered all the requirements from earlier in the conversation.",
            },
        ],
    },
    {
        "id": "ml_pipeline",
        "label": "Build an ML training pipeline (multi-step)",
        "description": (
            "Design and configure an end-to-end ML training pipeline for a text classifier: "
            "data loading, preprocessing, training, evaluation, and deployment. 5 steps."
        ),
        "turns": [
            {
                "turn": 1,
                "action": "Define data requirements",
                "wiki_writes": [
                    WikiEntry("data.source",  "Dataset: 50k labelled support tickets (text, label) from GCS bucket gs://ml-data/tickets", "fact", 1, 0.99, "agent_7b"),
                    WikiEntry("data.split",   "Train 80% / Val 10% / Test 10% — stratified by label", "plan", 1, 0.95, "agent_7b"),
                    WikiEntry("plan.step1",   "Preprocess: lowercase, remove PII, tokenize with BERT tokenizer", "plan", 1, 0.92, "agent_7b"),
                ],
                "agent_7b_answer": "Data spec written to wiki: 50k tickets, stratified 80/10/10 split, BERT tokenizer.",
                "agent_21b_answer": "We'll need data. Probably from somewhere. Let me think about how to split it.",
            },
            {
                "turn": 2,
                "action": "Design preprocessing pipeline",
                "wiki_reads": ["data.source", "plan.step1"],
                "wiki_writes": [
                    WikiEntry("preprocess.steps", "1.lowercase 2.regex PII removal (email,phone,SSN) 3.BERT tokenizer max_len=512 4.class weights for imbalance", "fact", 2, 0.96, "agent_7b"),
                    WikiEntry("preprocess.imbalance", "Label distribution: 60% class-0, 30% class-1, 10% class-2 — use class_weight='balanced'", "fact", 2, 0.95, "agent_7b"),
                ],
                "agent_7b_answer": "Preprocessing pipeline defined in wiki. Class imbalance noted — balanced weights configured.",
                "agent_21b_answer": "BERT tokenizer sounds right. We might want to handle class imbalance but I'm not sure of the distribution.",
            },
            {
                "turn": 3,
                "action": "Configure model & training",
                "wiki_reads": ["preprocess.steps", "preprocess.imbalance"],
                "wiki_writes": [
                    WikiEntry("model.arch",   "bert-base-uncased + linear head (768→3), dropout=0.1", "fact", 3, 0.97, "agent_7b"),
                    WikiEntry("train.config", "lr=2e-5, batch=32, epochs=5, warmup=0.1, AdamW, fp16", "fact", 3, 0.96, "agent_7b"),
                    WikiEntry("train.metric", "Primary: macro-F1. Secondary: per-class F1", "fact", 3, 0.98, "agent_7b"),
                ],
                "agent_7b_answer": "Model + training config stored in wiki. BERT fine-tune with balanced weights from turn 2.",
                "agent_21b_answer": "BERT fine-tuning with standard hyperparameters. Learning rate around 2e-5 is typical.",
            },
            {
                "turn": 4,
                "action": "Define evaluation & thresholds",
                "wiki_reads": ["train.metric", "preprocess.imbalance"],
                "wiki_writes": [
                    WikiEntry("eval.thresholds", "Gate: macro-F1 ≥ 0.82 on val; deploy only if test-macro-F1 ≥ 0.80", "fact", 4, 0.97, "agent_7b"),
                    WikiEntry("eval.per_class",  "Min class-2 F1 ≥ 0.70 (rarest class — most business-critical)", "fact", 4, 0.96, "agent_7b"),
                ],
                "agent_7b_answer": "Evaluation thresholds set in wiki using imbalance info from turn 2. Class-2 guarded separately.",
                "agent_21b_answer": "Good F1 scores are above 0.8. We should check per-class metrics too.",
            },
            {
                "turn": 5,
                "action": "Deployment plan",
                "wiki_reads": ["model.arch", "train.config", "eval.thresholds", "data.source"],
                "wiki_writes": [
                    WikiEntry("deploy.plan", "FastAPI /predict endpoint → Docker → GKE. Model weights: GCS gs://ml-models/ticket-clf/v1/", "conclusion", 5, 0.97, "agent_7b"),
                    WikiEntry("deploy.monitor", "Monitor: inference latency P99, macro-F1 on shadow traffic, data drift via Evidently", "conclusion", 5, 0.95, "agent_7b"),
                ],
                "agent_7b_answer": (
                    "Full pipeline assembled from wiki: BERT fine-tune (turn 3 config) → "
                    "F1 ≥ 0.82 gate (turn 4) → FastAPI+Docker+GKE deploy → drift monitoring."
                ),
                "agent_21b_answer": (
                    "Deploy with FastAPI and Docker. Monitor performance. "
                    "I think we covered most requirements."
                ),
            },
        ],
    },
]

# ---------------------------------------------------------------------------
# Benchmark data: small+wiki vs large (no wiki)
# ---------------------------------------------------------------------------

BENCHMARK_ROWS: list[dict] = [
    {
        "task_type": "Multi-step debugging (5 turns)",
        "small_wiki_score": 94,
        "large_no_wiki_score": 61,
        "metric": "Task completion %",
        "note": "Small agent recalls every prior finding via wiki; large model forgets early turns.",
    },
    {
        "task_type": "Long-horizon planning (10 turns)",
        "small_wiki_score": 89,
        "large_no_wiki_score": 52,
        "metric": "Plan coherence score",
        "note": "Wiki externalises plan steps so they can't be overwritten by attention drift.",
    },
    {
        "task_type": "Fact-intensive QA (requires 8+ facts)",
        "small_wiki_score": 91,
        "large_no_wiki_score": 74,
        "metric": "Accuracy %",
        "note": "Large model holds more in-context facts but loses track of interdependencies.",
    },
    {
        "task_type": "Cross-turn consistency",
        "small_wiki_score": 97,
        "large_no_wiki_score": 63,
        "metric": "Contradiction-free turns %",
        "note": "Wiki enforces a single source of truth; large model contradicts itself across turns.",
    },
    {
        "task_type": "Token efficiency (same task)",
        "small_wiki_score": 100,
        "large_no_wiki_score": 310,
        "metric": "Relative token cost (lower = better)",
        "note": "Small model context stays lean; large model must re-read full history.",
        "lower_is_better": True,
    },
]

# ---------------------------------------------------------------------------
# Wiki mechanics explainer
# ---------------------------------------------------------------------------

WIKI_CONCEPTS: list[dict] = [
    {
        "title": "Read operation",
        "icon": "📖",
        "detail": (
            "The agent queries the wiki by key before generating each response. "
            "This replaces re-reading the full conversation history — O(1) lookup "
            "vs O(n) context scan."
        ),
        "analogy": "Like grep on a structured notebook rather than scrolling a chat log.",
        "code": "wiki.read('root_cause.io')  →  'PaymentGateway.verify() blocks 1.9 s'",
    },
    {
        "title": "Write operation",
        "icon": "✍️",
        "detail": (
            "After each turn the agent writes new facts, observations, and updated plans "
            "to the wiki. The wiki is the only persistent store — the agent's context window "
            "can be reset between turns without losing information."
        ),
        "analogy": "Like a developer committing notes to a shared doc after each investigation step.",
        "code": "wiki.write('fix.code', 'asyncio.wait_for(..., timeout=0.5)', kind='fact')",
    },
    {
        "title": "Entry kinds",
        "icon": "🗂️",
        "detail": (
            "Entries are typed: **fact** (observed ground truth), **plan** (next steps), "
            "**observation** (intermediate finding), **conclusion** (final result). "
            "Typed entries let agents filter by kind to reconstruct a reasoning chain."
        ),
        "analogy": "Like a lab notebook separating hypotheses from measurements from conclusions.",
        "code": "wiki.query(kind='plan')  →  list of all open plan entries",
    },
    {
        "title": "Confidence scores",
        "icon": "📊",
        "detail": (
            "Each entry carries a confidence score (0–1). Low-confidence entries trigger "
            "a verification sub-task before they propagate to conclusions. "
            "This mirrors how scientists distinguish hypothesis from confirmed result."
        ),
        "analogy": "Like a flag system — yellow (hypothesis), green (verified), red (retracted).",
        "code": "wiki.read_if_confident('root_cause.io', threshold=0.90)",
    },
    {
        "title": "Why small beats large",
        "icon": "⚡",
        "detail": (
            "A 7B model with a wiki outperforms a 21B model without one because: "
            "(1) the wiki externalises memory so context stays free for reasoning; "
            "(2) facts are precise, not reconstructed from attention; "
            "(3) plans are explicit, not implicit in the token stream. "
            "Architecture beats parameter count when the bottleneck is memory, not reasoning."
        ),
        "analogy": "Like a junior engineer with a thorough personal wiki outperforming a senior who relies on memory alone.",
        "code": "# Context window budget:\n# 21B no-wiki: 80% history, 20% reasoning\n# 7B + wiki: 15% wiki reads, 85% reasoning",
    },
]

# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def simulate_task(task_id: str, turn_idx: int) -> dict:
    """
    Return the state of the wiki and both agent answers after `turn_idx + 1` turns.
    """
    task = next(t for t in TASKS if t["id"] == task_id)
    wiki_state: list[WikiEntry] = []
    answers_7b: list[str] = []
    answers_21b: list[str] = []

    for i, turn in enumerate(task["turns"]):
        if i > turn_idx:
            break
        wiki_state.extend(turn.get("wiki_writes", []))
        answers_7b.append(f"**Turn {turn['turn']} ({turn['action']}):** {turn['agent_7b_answer']}")
        answers_21b.append(f"**Turn {turn['turn']} ({turn['action']}):** {turn['agent_21b_answer']}")

    return {
        "task": task,
        "turn_idx": turn_idx,
        "wiki_state": wiki_state,
        "answers_7b": answers_7b,
        "answers_21b": answers_21b,
    }


def format_wiki_state(entries: list[WikiEntry]) -> str:
    if not entries:
        return "_Wiki is empty._"
    kind_icons = {"fact": "📌", "plan": "📋", "observation": "🔍", "conclusion": "✅"}
    lines = ["| Key | Kind | Value | Confidence |", "|-----|------|-------|------------|"]
    for e in entries:
        icon = kind_icons.get(e.kind, "📄")
        conf_bar = "█" * int(e.confidence * 10) + "░" * (10 - int(e.confidence * 10))
        lines.append(f"| `{e.key}` | {icon} {e.kind} | {e.value[:60]}{'…' if len(e.value) > 60 else ''} | {conf_bar} {e.confidence:.2f} |")
    return "\n".join(lines)


def format_comparison(state: dict) -> tuple[str, str, str]:
    """Return (wiki_md, agent_7b_md, agent_21b_md)."""
    wiki_md = format_wiki_state(state["wiki_state"])
    agent_7b_md = "\n\n".join(state["answers_7b"]) or "_No turns yet._"
    agent_21b_md = "\n\n".join(state["answers_21b"]) or "_No turns yet._"
    return wiki_md, agent_7b_md, agent_21b_md


def get_benchmark_chart_data() -> tuple[list[str], list[float], list[float]]:
    labels = [r["task_type"] for r in BENCHMARK_ROWS if not r.get("lower_is_better")]
    small_wiki = [r["small_wiki_score"] for r in BENCHMARK_ROWS if not r.get("lower_is_better")]
    large_no_wiki = [r["large_no_wiki_score"] for r in BENCHMARK_ROWS if not r.get("lower_is_better")]
    return labels, small_wiki, large_no_wiki
