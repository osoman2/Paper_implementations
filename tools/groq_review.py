#!/usr/bin/env python3
"""
Reviewer agent for poc/* pull requests, run inside GitHub Actions with Groq (free tier).

    GROQ_API_KEY=... python tools/groq_review.py --base origin/deploy \
        --out .agent-review

Inputs it gives the model: REVIEW_CHECKLIST.md, the deterministic scope report, the PR body, and the
diff (new POC module in full, the new tab block, the INBOX entry). Output:
  .agent-review/verdict.json   {"verdict": "PASS"|"FAIL", "failed_items": [...], "summary": "..."}
  .agent-review/comment.md     the PR comment ("Agent review: PASS|FAIL" + item table + fixes)

The model never sees the authoring session and cannot push or change statuses: the workflow turns
the verdict into the job result. Exit code is always 0 unless the API call itself fails (then 2),
so the workflow decides what a failure means.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parent.parent
ROOT = TOOLS_ROOT  # repository whose diff is reviewed; --repo points it at the PR checkout
API = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_DIFF_CHARS = 60_000  # keeps one request inside free-tier token limits

SYSTEM = """You are a strict code reviewer for a public demo site. You did NOT write this code.
Review the pull request against the checklist. Be concrete: cite file:line from the diff.
Judge only what the diff shows; do not invent problems you cannot point to.
Return ONLY a JSON object with this shape:
{
  "items": [{"n": <item number>, "result": "PASS"|"FAIL"|"N/A", "ref": "<file:line or ''>",
             "note": "<one sentence>", "fix": "<exact fix if FAIL, else ''>"}],
  "verdict": "PASS"|"FAIL",
  "summary": "<one sentence>"
}
verdict is PASS only if every item is PASS or N/A. Cover every checklist item 1-12."""


def _error(msg: str) -> None:
    """Print an error; inside GitHub Actions also as an annotation, readable without the logs."""
    msg = str(msg)
    print(msg, file=sys.stderr)
    if os.environ.get("GITHUB_ACTIONS"):
        print("::error::" + msg.replace("%", "%25").replace("\r", "").replace("\n", "%0A"))


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=True, cwd=ROOT).stdout


def collect_diff(base: str) -> str:
    files = [l.split("\t") for l in sh("git", "diff", "--name-status", f"{base}...HEAD").splitlines()]
    parts = []
    # New files in full (that is where the POC lives), modified files as unified diff.
    for status, *paths in files:
        path = paths[-1]
        if status.startswith("A") and path.endswith((".py", ".md", ".txt", ".json")):
            body = (ROOT / path).read_text(encoding="utf-8", errors="ignore")
            numbered = "\n".join(f"{i:4d}  {l}" for i, l in enumerate(body.splitlines(), 1))
            parts.append(f"=== NEW FILE {path} ===\n{numbered}")
    parts.append("=== DIFF OF MODIFIED FILES ===\n" + sh("git", "diff", "-U3", "--diff-filter=M", f"{base}...HEAD"))
    text = "\n\n".join(parts)
    if len(text) > MAX_DIFF_CHARS:
        text = text[:MAX_DIFF_CHARS] + f"\n\n[... truncated {len(text) - MAX_DIFF_CHARS} chars ...]"
    return text


def call_groq(messages: list[dict], model: str, key: str) -> dict:
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": 0, "response_format": {"type": "json_object"}}
    ).encode()
    for attempt in range(4):
        req = urllib.request.Request(
            API,
            data=payload,
            # Groq sits behind Cloudflare, which rejects the default Python-urllib user agent.
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "User-Agent": "vrl-agent-review/1.0 (+https://github.com/osoman2/Paper_implementations)",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                content = json.load(r)["choices"][0]["message"]["content"]
                return json.loads(content)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:  # free-tier rate limit: wait and retry
                wait = int(e.headers.get("retry-after", "20") or 20)
                time.sleep(min(max(wait, 5), 60))
                continue
            raise RuntimeError(f"Groq HTTP {e.code}: {e.read()[:400].decode('utf-8', 'replace')}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Groq unreachable: {e.reason}") from e
        except json.JSONDecodeError as e:
            if attempt < 3:
                continue
            raise RuntimeError(f"Groq returned non-JSON: {e}") from e
    raise RuntimeError("Groq: retries exhausted")


def render(review: dict, model: str) -> str:
    verdict = review.get("verdict", "FAIL")
    rows = ["| # | Result | Where | Note |", "|---|---|---|---|"]
    fixes = []
    for it in sorted(review.get("items", []), key=lambda x: x.get("n", 0)):
        note = str(it.get("note", "")).replace("|", "\\|")
        rows.append(f"| {it.get('n')} | {it.get('result')} | `{it.get('ref') or '-'}` | {note} |")
        if it.get("result") == "FAIL" and it.get("fix"):
            fixes.append(f"- **Item {it.get('n')}:** {it['fix']}")
    out = [f"## Agent review: {verdict}", "", review.get("summary", ""), "", *rows]
    if fixes:
        out += ["", "### Fixes", *fixes]
    out += ["", f"<sub>Independent reviewer · {model} via Groq · checklist: REVIEW_CHECKLIST.md</sub>"]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default=".agent-review")
    ap.add_argument("--pr-body", default="")
    ap.add_argument("--scope", default="")
    ap.add_argument("--repo", default="", help="checkout of the PR head (default: this repository)")
    ap.add_argument("--checklist", default="", help="checklist file (default: REVIEW_CHECKLIST.md next to this tool)")
    args = ap.parse_args()
    global ROOT
    if args.repo:
        ROOT = Path(args.repo).resolve()

    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        _error("GROQ_API_KEY missing")
        return 2
    model = os.environ.get("GROQ_MODEL") or DEFAULT_MODEL
    # The checklist comes from the trusted copy, never from the PR under review.
    checklist = Path(args.checklist or TOOLS_ROOT / "REVIEW_CHECKLIST.md").read_text(encoding="utf-8")
    scope = Path(args.scope).read_text(encoding="utf-8") if args.scope and Path(args.scope).exists() else ""
    body = Path(args.pr_body).read_text(encoding="utf-8") if args.pr_body and Path(args.pr_body).exists() else ""
    user = (
        f"# Checklist\n{checklist}\n\n# Deterministic scope check\n{scope}\n\n"
        f"# PR description\n{body[:4000]}\n\n# Code\n{collect_diff(args.base)}"
    )
    try:
        review = call_groq([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], model, key)
    except RuntimeError as e:
        _error(f"Reviewer could not run: {e}")
        return 2

    items = review.get("items", [])
    failed = [it.get("n") for it in items if it.get("result") == "FAIL"]
    # Never trust a PASS that contradicts its own items.
    verdict = "FAIL" if failed or review.get("verdict") != "PASS" else "PASS"
    review["verdict"] = verdict
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "verdict.json").write_text(
        json.dumps({"verdict": verdict, "failed_items": failed, "summary": review.get("summary", "")}, indent=2)
    )
    (out / "comment.md").write_text(render(review, model), encoding="utf-8")
    print(f"verdict {verdict}; failed items {failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
