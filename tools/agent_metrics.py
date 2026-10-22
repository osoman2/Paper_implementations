#!/usr/bin/env python3
"""
Weekly metrics for the author / reviewer / CI loop, from the GitHub REST API.

    GITHUB_TOKEN=... python tools/agent_metrics.py            # print the table
    GITHUB_TOKEN=... python tools/agent_metrics.py --write    # also write METRICS.md

Per ISO week:
  - agent PRs opened (head branch poc/*)
  - PRs where the reviewer agent failed at least once ("Agent review: FAIL" comment)
  - PRs where the smoke test failed at least once (check run "smoke" on any PR commit)
  - merged PRs and median hours from open to merge
  - deploys that failed after merge (Deploy Hugging Face Space runs on the deploy branch; target 0)

Token: GITHUB_TOKEN or GH_TOKEN, or the output of `gh auth token`.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = os.environ.get("GITHUB_REPOSITORY", "osoman2/Paper_implementations")
DEPLOY_BRANCH = "claude/implement-hybrid-search-WEN8O"
AGENT_PREFIX = "poc/"
METRICS_MD = Path(__file__).resolve().parent.parent / "METRICS.md"


def _token() -> str:
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        return tok
    try:
        return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _get(path: str, params: dict | None = None):
    url = f"https://api.github.com{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
    tok = _token()
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _paged(path: str, params: dict, key: str | None = None, limit: int = 1000) -> list:
    out, page = [], 1
    while len(out) < limit:
        data = _get(path, {**params, "per_page": 100, "page": page})
        items = data[key] if key else data
        out += items
        if len(items) < 100:
            break
        page += 1
    return out


def _ts(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def _week(d: datetime) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def collect(weeks: int) -> dict[str, dict]:
    since = datetime.now(timezone.utc) - timedelta(weeks=weeks)
    stats: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    merge_hours: dict[str, list[float]] = defaultdict(list)

    prs = _paged(f"/repos/{REPO}/pulls", {"state": "all", "base": DEPLOY_BRANCH, "sort": "created", "direction": "desc"})
    for pr in prs:
        created = _ts(pr["created_at"])
        if created < since or not pr["head"]["ref"].startswith(AGENT_PREFIX):
            continue
        wk = _week(created)
        stats[wk]["agent_prs"] += 1
        n = pr["number"]
        comments = _paged(f"/repos/{REPO}/issues/{n}/comments", {})
        if any("Agent review: FAIL" in (c.get("body") or "") for c in comments):
            stats[wk]["review_failed"] += 1
        smoke_failed = False
        for commit in _paged(f"/repos/{REPO}/pulls/{n}/commits", {}):
            runs = _get(f"/repos/{REPO}/commits/{commit['sha']}/check-runs", {"check_name": "smoke"})["check_runs"]
            if any(r.get("conclusion") == "failure" for r in runs):
                smoke_failed = True
                break
        if smoke_failed:
            stats[wk]["ci_blocked"] += 1
        merged = _ts(pr.get("merged_at"))
        if merged:
            stats[wk]["merged"] += 1
            merge_hours[wk].append((merged - created).total_seconds() / 3600)

    runs = _paged(
        f"/repos/{REPO}/actions/workflows/deploy-space.yml/runs",
        {"branch": DEPLOY_BRANCH, "event": "push", "created": f">={since.date().isoformat()}"},
        key="workflow_runs",
    )
    for run in runs:
        wk = _week(_ts(run["created_at"]))
        stats[wk]["deploys"] += 1
        if run.get("conclusion") == "failure":
            stats[wk]["deploys_failed"] += 1

    for wk, hours in merge_hours.items():
        stats[wk]["median_hours_to_merge"] = round(statistics.median(hours), 1)
    return {wk: dict(v) for wk, v in sorted(stats.items(), reverse=True)}


COLUMNS = [
    ("agent_prs", "Agent PRs"),
    ("review_failed", "Reviewer FAIL"),
    ("ci_blocked", "Blocked by CI"),
    ("merged", "Merged"),
    ("median_hours_to_merge", "Median h to merge"),
    ("deploys", "Deploys"),
    ("deploys_failed", "Deploys failed (target 0)"),
]


def render(stats: dict[str, dict]) -> str:
    lines = [
        "# Agent loop metrics",
        "",
        f"Author agent → reviewer agent → CI gate → merge → deploy, for `{REPO}`.",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `tools/agent_metrics.py`.",
        "",
        "| Week | " + " | ".join(c[1] for c in COLUMNS) + " |",
        "|---|" + "---:|" * len(COLUMNS),
    ]
    for wk, s in stats.items():
        lines.append(f"| {wk} | " + " | ".join(str(s.get(k, 0 if k != 'median_hours_to_merge' else '—')) for k, _ in COLUMNS) + " |")
    if not stats:
        lines.append("| — | " + " | ".join("0" for _ in COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=8)
    ap.add_argument("--write", action="store_true", help=f"write {METRICS_MD.name}")
    args = ap.parse_args()
    try:
        stats = collect(args.weeks)
    except Exception as exc:  # noqa: BLE001
        print(f"GitHub API error: {exc}", file=sys.stderr)
        sys.exit(1)
    md = render(stats)
    print(md)
    if args.write:
        METRICS_MD.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
