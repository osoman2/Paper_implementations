#!/usr/bin/env python3
"""
Candidate feeds besides the newsletters: Hugging Face Daily Papers and new GitHub repos with fast stars.

    python tools/source_feeds.py fetch --out candidates.json   # GitHub Action (source-feeds.yml)
    python tools/source_feeds.py show --since 2026-10-28       # daily author agent, step 3

Agent sandboxes cannot call the GitHub search API, so the Action fetches both feeds every day and
publishes candidates.json on the `feeds` branch; `show` reads it from there with plain git. If the
branch is missing, `show` falls back to fetching Hugging Face live (GitHub needs the Action).

Pre-filter (cheap, before any scoring): Hugging Face papers need community upvotes, GitHub repos need
stars and an AI topic. Only the top items of each feed are kept so the agent scores a short list.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

HF_API = "https://huggingface.co/api/daily_papers"
GH_API = "https://api.github.com/search/repositories"
HF_MIN_UPVOTES = 20
HF_TOP_PER_DAY = 8
GH_MIN_STARS = 150
GH_TOP = 12
GH_TOPICS = ("llm", "rag", "ai-agents", "agents", "retrieval", "mcp", "llm-inference", "vector-search")
DAYS = 7


def _get(url: str, headers: dict | None = None) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "paper-implementations-feeds", **(headers or {})})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def huggingface(days: int = DAYS) -> list[dict]:
    """Top community-voted papers of each of the last `days` days. Links go to arXiv, the original source."""
    out = []
    today = datetime.now(timezone.utc).date()
    for i in range(days):
        day = (today - timedelta(days=i)).isoformat()
        try:
            items = _get(f"{HF_API}?date={day}&limit=100")
        except Exception as err:  # one missing day must not drop the whole feed
            print(f"[hf] {day}: {err}", file=sys.stderr)
            continue
        papers = [it["paper"] for it in items if it.get("paper", {}).get("upvotes", 0) >= HF_MIN_UPVOTES]
        papers.sort(key=lambda p: p.get("upvotes", 0), reverse=True)
        for p in papers[:HF_TOP_PER_DAY]:
            out.append(
                {
                    "source": "huggingface",
                    "date": day,
                    "title": " ".join(p.get("title", "").split()),
                    "url": f"https://arxiv.org/abs/{p['id']}",
                    "repo": p.get("githubRepo") or "",
                    "signal": f"{p.get('upvotes', 0)} upvotes",
                    "summary": (p.get("ai_summary") or p.get("summary") or "")[:600],
                }
            )
    return out


def github(days: int = DAYS) -> list[dict]:
    """New repos (created in the last `days` days) with an AI topic, ranked by stars. Needs GITHUB_TOKEN."""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("[github] no GITHUB_TOKEN: skipped (runs in the source-feeds Action)", file=sys.stderr)
        return []
    since = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
    seen: dict[str, dict] = {}
    for topic in GH_TOPICS:
        q = f"topic:{topic} created:>={since} stars:>={GH_MIN_STARS}"
        url = f"{GH_API}?q={urllib.parse.quote(q)}&sort=stars&order=desc&per_page=20"
        try:
            data = _get(url, {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
        except Exception as err:
            print(f"[github] {topic}: {err}", file=sys.stderr)
            continue
        for r in data.get("items", []):
            if r.get("fork") or r["full_name"] in seen:
                continue
            seen[r["full_name"]] = {
                "source": "github",
                "date": r["created_at"][:10],
                "title": f"{r['full_name']}: {r.get('description') or ''}".strip().rstrip(":"),
                "url": r["html_url"],
                "repo": r["html_url"],
                "signal": f"{r['stargazers_count']} stars",
                "stars": r["stargazers_count"],
                "summary": (r.get("description") or "")[:600],
            }
    ranked = sorted(seen.values(), key=lambda r: r["stars"], reverse=True)[:GH_TOP]
    for r in ranked:
        r.pop("stars")
    return ranked


def fetch(out_path: str) -> int:
    items = huggingface() + github()
    payload = {"fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "items": items}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    counts = {s: sum(1 for i in items if i["source"] == s) for s in ("huggingface", "github")}
    print(f"wrote {len(items)} candidates to {out_path}: {counts}")
    return 0


def _from_feeds_branch() -> dict | None:
    try:
        subprocess.run(["git", "fetch", "-q", "origin", "feeds"], check=True, capture_output=True, timeout=60)
        raw = subprocess.run(
            ["git", "show", "FETCH_HEAD:candidates.json"], check=True, capture_output=True, text=True
        ).stdout
        return json.loads(raw)
    except Exception as err:
        print(f"[show] feeds branch not readable ({err}); fetching Hugging Face live", file=sys.stderr)
        return None


def show(since: str) -> int:
    payload = _from_feeds_branch() or {"fetched_at": "live", "items": huggingface()}
    items = [i for i in payload["items"] if i["date"] > since]
    print(f"# Feed candidates after {since} (fetched {payload['fetched_at']}): {len(items)}\n")
    for i in items:
        repo = f" · code: {i['repo']}" if i["repo"] and i["repo"] != i["url"] else ""
        print(f"- [{i['source']} · {i['date']} · {i['signal']}] {i['title']}\n  {i['url']}{repo}\n  {i['summary']}\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="fetch both feeds into a JSON file (GitHub Action)")
    f.add_argument("--out", default="candidates.json")
    s = sub.add_parser("show", help="list candidates newer than a date (author agent)")
    s.add_argument("--since", default=(date.today() - timedelta(days=1)).isoformat(), help="YYYY-MM-DD, exclusive")
    args = ap.parse_args()
    return fetch(args.out) if args.cmd == "fetch" else show(args.since)


if __name__ == "__main__":
    sys.exit(main())
