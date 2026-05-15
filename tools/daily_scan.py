#!/usr/bin/env python3
"""
JournalClub.io daily paper scanner.

Reads new emails from noreply@newsletter.journalclub.io, scores each paper's
relevance for AI engineering (RAG, LLMs, retrieval, agents) using Claude Haiku,
and writes a ranked report to INBOX.md.

State: tools/.seen_ids.json tracks processed email IDs so only new emails are
scored on each run.

Exit codes:
  0 — no new relevant papers (score >= 6)
  2 — new high-relevance papers found (triggers asyncRewake notification)
"""

import json
import os
import sys
import base64
import re
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime, date

try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

try:
    from groq import Groq
    _HAS_GROQ = True
except ImportError:
    _HAS_GROQ = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).parent.parent
STATE_FILE   = Path(__file__).parent / ".seen_ids.json"
INBOX_MD     = REPO_ROOT / "INBOX.md"
SENDER       = "noreply@newsletter.journalclub.io"
RELEVANCE_THRESHOLD = 7   # papers scoring >= this trigger asyncRewake notification
GMAIL_CREDS  = Path.home() / ".gmail-mcp" / "token.json"

# ---------------------------------------------------------------------------
# Gmail helpers (direct API — MCP not available in subprocess context)
# ---------------------------------------------------------------------------

def _gmail_token() -> str:
    # Cloud/routine environment: credentials via env vars
    if os.environ.get("GMAIL_REFRESH_TOKEN"):
        creds = {
            "client_id":     os.environ["GMAIL_CLIENT_ID"],
            "client_secret": os.environ["GMAIL_CLIENT_SECRET"],
            "refresh_token": os.environ["GMAIL_REFRESH_TOKEN"],
        }
    else:
        with open(GMAIL_CREDS) as f:
            creds = json.load(f)
    data = urllib.parse.urlencode({
        "client_id": creds["client_id"],
        "client_secret": creds["client_secret"],
        "refresh_token": creds["refresh_token"],
        "grant_type": "refresh_token",
    }).encode()
    req = urllib.request.Request(
        "https://oauth2.googleapis.com/token", data=data, method="POST"
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)["access_token"]


def _gmail_get(path: str, token: str) -> dict:
    url = f"https://gmail.googleapis.com/gmail/v1/users/me/{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def fetch_email_ids(token: str) -> list[str]:
    q = urllib.parse.quote(f"from:{SENDER}")
    resp = _gmail_get(f"messages?q={q}&maxResults=50", token)
    return [m["id"] for m in resp.get("messages", [])]


def _decode_b64(data: str) -> str:
    pad = 4 - len(data) % 4
    if pad != 4:
        data += "=" * pad
    return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")


def _extract_html(payload: dict) -> str:
    if payload.get("mimeType") == "text/html":
        return _decode_b64(payload.get("body", {}).get("data", ""))
    for part in payload.get("parts", []):
        result = _extract_html(part)
        if result:
            return result
    return ""


def _strip_html(html: str) -> str:
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL)
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL)
    html = re.sub(r"<br\s*/?>", "\n", html)
    html = re.sub(r"<[^>]+>", " ", html)
    for ent, rep in [("&amp;", "&"), ("&#39;", "'"), ("&quot;", '"'),
                     ("&lt;", "<"), ("&gt;", ">")]:
        html = html.replace(ent, rep)
    html = re.sub(r"[ \t]+", " ", html)
    html = re.sub(r"\n\s*\n\s*\n+", "\n\n", html)
    return html.strip()


def fetch_email_content(msg_id: str, token: str) -> dict:
    msg = _gmail_get(f"messages/{msg_id}?format=full", token)
    subject = next(
        (h["value"] for h in msg["payload"].get("headers", []) if h["name"] == "Subject"),
        "No subject",
    )
    html = _extract_html(msg.get("payload", {}))
    body = _strip_html(html)[:3000]
    return {"id": msg_id, "subject": subject, "body": body}

# ---------------------------------------------------------------------------
# LLM relevance scoring
# ---------------------------------------------------------------------------

RELEVANCE_PROMPT = """\
You are a strict relevance filter for an AI engineer who builds production systems: \
RAG pipelines, LLM applications, hybrid search, AI agents, embeddings, and LLM inference optimisation.

Your job is to score papers for DIRECT implementability, not vague conceptual similarity.
Be conservative. A paper about recommendation systems is NOT automatically relevant to RAG.
A paper using "AI" or "deep learning" for an unrelated domain (weather, finance, biology, traffic) scores 1–3.

CALIBRATION EXAMPLES (use these as anchors):
- "Query-adaptive hybrid search with IDF-based alpha" → 10 (literally what RAG engineers tune)
- "CAD score: semantic diversity metric for LLM outputs" → 9 (direct LLM evaluation tool)
- "Speculative decoding for faster LLM inference" → 10 (core inference optimisation)
- "ColBERT late interaction for passage retrieval" → 9 (retrieval architecture)
- "Isolation Forest improvement for anomaly detection" → 5 (generic ML, not LLM-specific)
- "Recommendation system with transformer+clustering" → 4 (uses transformers, but not for LLMs)
- "RL for financial portfolio management" → 2 (RL is ML, but domain is finance, not AI systems)
- "Precipitation forecasting with AI weather models" → 1 (domain application, not AI engineering)
- "Traffic light control with deep learning" → 1 (domain application)
- "Pine wilt disease detection with CNNs" → 1 (domain application)

SCORING RUBRIC:
8–10: Must implement a technique directly usable in: hybrid search, RAG chunking/retrieval/reranking,
      LLM output evaluation, prompt optimisation, embedding models, agent memory, inference speed/cost.
      No adaptation needed — drop-in applicable.
5–7:  Technique is applicable with some adaptation. Must use LLMs, embeddings, or retrieval as a
      core component (not just "uses transformers for X unrelated task").
1–4:  Generic ML, domain-specific AI, or no LLM/retrieval component.

Paper title: {title}

Paper content:
{body}

Respond with ONLY valid JSON (no markdown fences, no extra text):
{{"score": <int 1-10>, "reason": "<one crisp sentence on why this score, not just rephrasing the title>", "implement": <true if score>=8>, "poc_idea": "<concrete one-sentence POC description, or empty string if score<7>"}}"""


def score_paper(title: str, body: str) -> dict:
    prompt = RELEVANCE_PROMPT.format(title=title, body=body[:1500])

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    groq_key      = os.environ.get("GROQ_API_KEY")

    if _HAS_ANTHROPIC and anthropic_key:
        client = anthropic.Anthropic(api_key=anthropic_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
    elif _HAS_GROQ and groq_key:
        client = Groq(api_key=groq_key)
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = completion.choices[0].message.content.strip()
    else:
        raise RuntimeError("No LLM API key found (set ANTHROPIC_API_KEY or GROQ_API_KEY)")

    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return json.loads(text)

# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        data = json.loads(STATE_FILE.read_text())
        # backwards compat: old format was just a list of IDs
        if isinstance(data, list):
            return {"seen_ids": data, "last_run_date": None}
        return data
    return {"seen_ids": [], "last_run_date": None}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def already_ran_today(state: dict) -> bool:
    return state.get("last_run_date") == date.today().isoformat()

# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_inbox(results: list[dict], new_count: int) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# INBOX — JournalClub.io Paper Scan",
        f"Last updated: {now} · {new_count} new paper(s) scored this run\n",
        "## New Papers (this run)\n",
    ]

    new_papers = [r for r in results if r.get("is_new")]
    if not new_papers:
        lines.append("_No new papers since last scan._\n")
    else:
        new_papers.sort(key=lambda x: x["score"], reverse=True)
        for p in new_papers:
            badge = "🔴" if p["score"] >= 8 else "🟡" if p["score"] >= 6 else "⚪"
            lines.append(f"### {badge} [{p['subject']}] — Score {p['score']}/10")
            lines.append(f"> {p['reason']}")
            if p.get("poc_idea"):
                lines.append(f"\n**POC idea:** {p['poc_idea']}")
            lines.append("")

    # Previously seen high-relevance papers (persistent backlog)
    if INBOX_MD.exists():
        existing = INBOX_MD.read_text()
        backlog_start = existing.find("## Previously Tracked")
        if backlog_start != -1:
            lines.append(existing[backlog_start:])
            INBOX_MD.write_text("\n".join(lines))
            return

    # First run — build backlog header
    lines.append("## Previously Tracked\n")
    lines.append("_Papers from earlier scans are appended here automatically._\n")
    INBOX_MD.write_text("\n".join(lines))


def append_to_backlog(results: list[dict]) -> None:
    """Append high-relevance new papers to the backlog section."""
    if not INBOX_MD.exists():
        return
    high = [r for r in results if r.get("is_new") and r["score"] >= RELEVANCE_THRESHOLD]
    if not high:
        return
    content = INBOX_MD.read_text()
    additions = []
    for p in sorted(high, key=lambda x: x["score"], reverse=True):
        badge = "🔴" if p["score"] >= 8 else "🟡"
        additions.append(f"- {badge} **{p['subject']}** (score {p['score']}/10) — {p['reason']}")
    insertion = "\n".join(additions) + "\n"
    content = content.replace(
        "_Papers from earlier scans are appended here automatically._\n",
        "_Papers from earlier scans are appended here automatically._\n\n" + insertion,
        1,
    )
    INBOX_MD.write_text(content)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    force = "--force" in sys.argv

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GROQ_API_KEY")):
        print("[scan] No LLM API key found (ANTHROPIC_API_KEY or GROQ_API_KEY) — skipping.", file=sys.stderr)
        return 0

    state = load_state()

    if not force and already_ran_today(state):
        print(f"[scan] Already ran today ({state['last_run_date']}). Use --force to override.", file=sys.stderr)
        return 0

    print("[scan] Authenticating with Gmail…", file=sys.stderr)
    try:
        token = _gmail_token()
    except Exception as e:
        print(f"[scan] Gmail auth failed: {e}", file=sys.stderr)
        return 0

    print("[scan] Fetching email IDs…", file=sys.stderr)
    all_ids = fetch_email_ids(token)
    seen    = set(state["seen_ids"])
    new_ids = [i for i in all_ids if i not in seen]

    if not new_ids:
        print("[scan] No new emails.", file=sys.stderr)
        state["last_run_date"] = date.today().isoformat()
        save_state(state)
        return 0

    print(f"[scan] {len(new_ids)} new email(s) to process.", file=sys.stderr)

    results = []
    high_relevance_found = False

    for msg_id in new_ids:
        try:
            email = fetch_email_content(msg_id, token)
            print(f"[scan] Scoring: {email['subject'][:60]}…", file=sys.stderr)
            assessment = score_paper(email["subject"], email["body"])
            results.append({
                "is_new": True,
                "subject": email["subject"],
                "score": assessment["score"],
                "reason": assessment["reason"],
                "implement": assessment["implement"],
                "poc_idea": assessment.get("poc_idea", ""),
            })
            if assessment["score"] >= RELEVANCE_THRESHOLD:
                high_relevance_found = True
            seen.add(msg_id)
        except Exception as e:
            print(f"[scan] Error on {msg_id}: {e}", file=sys.stderr)

    state["seen_ids"] = sorted(seen)
    state["last_run_date"] = date.today().isoformat()
    save_state(state)
    write_inbox(results, len(new_ids))
    append_to_backlog(results)

    top = max(results, key=lambda x: x["score"]) if results else None
    if top:
        print(
            f"[scan] Done. Top paper: '{top['subject'][:50]}…' score={top['score']}/10",
            file=sys.stderr,
        )

    # Exit 2 = asyncRewake trigger: wakes Claude with a notification
    return 2 if high_relevance_found else 0


if __name__ == "__main__":
    sys.exit(main())
