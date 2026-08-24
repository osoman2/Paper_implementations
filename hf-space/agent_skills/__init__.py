"""
Agent Skills: What 8,100 Trial Records Reveal About Skills in AI Agents.

Based on: AlphaSignal deep-dive · 2026-08-23
"AI agent skills: Why they succeed — and what causes them to crash"
Study: 8,135 trial records across Terminal-Bench 2.0 and SkillsBench.

Core findings (empirical, directly applicable):
  1. Skills work as procedural ANCHORS, not knowledge repositories.
     Procedural anchoring accounts for 65.7% of successes.
     Explicit knowledge injection accounts for only 4.5%.
     Distilled runbook skill > raw workflow memory by +6.06 percentage points.

  2. Annotation is CRITICAL during skill distillation.
     With success/failure hints: 74.6% task success.
     Without hints (mixed batch, no annotation): 40.0%.
     Gap: 34.6 percentage points — skills baked from unannotated trajectories
     permanently encode hallucinations and inefficiencies.

  3. Catalog scaling degrades retrieval precision but not task success.
     Precision drops: 29.6% (5 skills) → 3.3% (100 skills).
     Task success remains stable: ~36-39% — because related non-ground-truth
     distractors provide enough partial procedural support.
     Real danger: semantic confusability (skills look identical in embedding space).
     Fix: two-level retrieval — domain bucket → strict trigger rule.

Applicable to: Any production agent system, RAG pipeline orchestration, multi-step
LLM workflows, tool-using agents, coding agents, agent frameworks.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Empirical data from the study (8,135 trial records)
# ---------------------------------------------------------------------------

# Success rates by execution mode (%)
EXECUTION_MODES = {
    "No Skill\n(Raw Execution)": 58.4,
    "Workflow Memory\n(Raw Logs)": 62.2,
    "Distilled Skill\n(Runbook)": 68.3,
}

# Annotation impact (%)
ANNOTATION_DATA = {
    "With Success/Failure\nAnnotations": 74.6,
    "Without Annotations\n(Mixed Batch)": 40.0,
}

# Procedural anchoring vs knowledge injection breakdown
SUCCESS_BREAKDOWN = {
    "Procedural Anchoring\n(runbook stabilizes execution)": 65.7,
    "Explicit Knowledge\nInjection (facts)": 4.5,
    "Other / Mixed\nFactors": 29.8,
}

# Catalog scale: skill pool size → retrieval precision (%)
CATALOG_SCALE = {
    5: 29.6,
    10: 18.4,
    20: 11.2,
    50: 6.8,
    100: 3.3,
}

# Task success remains stable despite precision drop (%)
CATALOG_SUCCESS = {
    5: 36.2,
    10: 37.1,
    20: 38.4,
    50: 37.8,
    100: 38.9,
}

# Environment failure rates (%) — with vs without skills
ENV_FAILURES = {
    "Raw Execution\n(no skill)": 5.3,
    "Distilled Skill\n(runbook)": 0.2,
}

# ---------------------------------------------------------------------------
# Concrete skill examples (runbook vs raw memory formats)
# ---------------------------------------------------------------------------

RAW_MEMORY_EXAMPLE = """\
Trial #1 (success):
  cd /app && npm install && node index.js —port 8080
  curl http://localhost:8080/health → 200
  Response time: 812ms (failed SLA)
  Tried adding cache: redis.set('key', val, {EX: 300})
  Re-ran: 743ms → still slow
  Added Promise.all for parallel fetches
  Response: 387ms ✓

Trial #2 (failure):
  npm install → ENOENT: package.json not found
  tried: cd src && npm install — still failed
  gave up

Trial #3 (success):
  Moved to /app/src first, then npm install
  node server.js —workers 4
  All tests pass
"""

DISTILLED_RUNBOOK_EXAMPLE = """\
## Skill: Node.js API Latency Fix

### Prerequisites
- Node.js project with async request handlers
- Redis available (optional, for caching)

### Steps
1. Navigate to project root (where package.json lives)
   `cd /app` or `cd /app/src` — check which has package.json first
2. Install dependencies: `npm install`
3. Identify serialized awaits in route handlers:
   - BAD:  `const a = await fetchA(); const b = await fetchB();`
   - GOOD: `const [a, b] = await Promise.all([fetchA(), fetchB()]);`
4. Convert all independent awaits to Promise.all
5. Start promises early, await late:
   `const pA = fetchA(); const pB = fetchB(); const [a,b] = await Promise.all([pA,pB]);`
6. Re-test: `curl http://localhost:<port>/health` — target < 800ms

### Success Criteria
Response time < 800ms under load test.

### Common Failures
- ENOENT: run `find . -name package.json` to locate project root
- Redis not available: skip caching step, Promise.all alone usually suffices
"""

# ---------------------------------------------------------------------------
# Two-level retrieval architecture demo
# ---------------------------------------------------------------------------

SKILL_CATALOG: list[dict[str, Any]] = [
    {
        "id": "be-01", "domain": "Backend",
        "trigger": 'error contains "ECONNREFUSED"',
        "title": "Connection Refused Debug",
        "description": "Diagnose and fix TCP connection refusals: check service status, firewall, port binding.",
    },
    {
        "id": "be-02", "domain": "Backend",
        "trigger": 'latency > 800ms AND async handler',
        "title": "API Latency Fix (Promise.all)",
        "description": "Convert serialized awaits to Promise.all; start promises early, await late.",
    },
    {
        "id": "be-03", "domain": "Backend",
        "trigger": 'error contains "ENOENT" AND node project',
        "title": "Node.js Project Root",
        "description": "Locate package.json with find, then cd before npm install.",
    },
    {
        "id": "fe-01", "domain": "Frontend",
        "trigger": 'React component AND prop undefined',
        "title": "React Prop Undefined Debug",
        "description": "Add defaultProps or optional chaining; check parent component pass-through.",
    },
    {
        "id": "fe-02", "domain": "Frontend",
        "trigger": 'CSS layout AND overflow hidden',
        "title": "CSS Overflow Fix",
        "description": "Use overflow-x: auto on container; avoid negative margin tricks.",
    },
    {
        "id": "db-01", "domain": "Database",
        "trigger": 'query timeout AND no index',
        "title": "Slow Query Index Tuning",
        "description": "EXPLAIN ANALYZE → identify seq scan → CREATE INDEX CONCURRENTLY.",
    },
    {
        "id": "db-02", "domain": "Database",
        "trigger": 'deadlock AND concurrent writes',
        "title": "Deadlock Resolution",
        "description": "Enforce consistent lock order; use SELECT FOR UPDATE SKIP LOCKED.",
    },
    {
        "id": "infra-01", "domain": "Infrastructure",
        "trigger": 'disk full AND /var/log',
        "title": "Log Rotation Emergency",
        "description": "journalctl --vacuum-size=500M; logrotate -f /etc/logrotate.conf",
    },
]


def two_level_retrieval(task_description: str, error_signal: str) -> dict[str, Any]:
    """Simulate two-level retrieval: domain bucket → strict trigger match."""

    task_lower = task_description.lower()
    error_lower = error_signal.lower()
    combined = task_lower + " " + error_lower

    # Level 1: domain bucket classification
    domain_scores: dict[str, int] = {"Backend": 0, "Frontend": 0, "Database": 0, "Infrastructure": 0}

    backend_kws = ["api", "server", "node", "express", "latency", "connection", "port", "npm", "async"]
    frontend_kws = ["react", "css", "component", "render", "ui", "dom", "prop", "layout"]
    db_kws = ["query", "sql", "database", "index", "deadlock", "postgres", "mysql", "table"]
    infra_kws = ["disk", "log", "docker", "kubernetes", "memory", "cpu", "deploy", "container"]

    for kw in backend_kws:
        if kw in combined:
            domain_scores["Backend"] += 1
    for kw in frontend_kws:
        if kw in combined:
            domain_scores["Frontend"] += 1
    for kw in db_kws:
        if kw in combined:
            domain_scores["Database"] += 1
    for kw in infra_kws:
        if kw in combined:
            domain_scores["Infrastructure"] += 1

    selected_domain = max(domain_scores, key=lambda d: domain_scores[d])
    domain_skills = [s for s in SKILL_CATALOG if s["domain"] == selected_domain]

    # Level 2: strict trigger match
    matched_skill = None
    partial_matches: list[dict[str, Any]] = []

    for skill in domain_skills:
        trigger_lower = skill["trigger"].lower()
        trigger_parts = [p.strip().strip('"') for p in trigger_lower.replace(" and ", "|").split("|")]
        full_match = all(
            any(tp in combined for tp in part.split())
            for part in trigger_parts
        )
        if full_match:
            matched_skill = skill
            break
        partial = any(
            any(tp in combined for tp in part.split())
            for part in trigger_parts
        )
        if partial:
            partial_matches.append(skill)

    return {
        "domain_scores": domain_scores,
        "selected_domain": selected_domain,
        "domain_skills": domain_skills,
        "matched_skill": matched_skill,
        "partial_matches": partial_matches,
    }
