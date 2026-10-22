#!/usr/bin/env python3
"""
Deterministic scope check for agent POC pull requests (REVIEW_CHECKLIST item 11).

    python tools/check_pr_scope.py --base origin/claude/implement-hybrid-search-WEN8O

Prints a markdown report and exits 1 if the diff touches files a POC PR should not touch.
Allowed: one new hf-space/<folder>/, hf-space/app.py and INBOX.md. Anything else is listed so
the reviewer can check the PR body justifies it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

ALWAYS_OK = {"hf-space/app.py", "INBOX.md"}
OK_PREFIXES: tuple[str, ...] = ()


def changed_files(base: str) -> list[tuple[str, str]]:
    out = subprocess.run(
        ["git", "diff", "--name-status", f"{base}...HEAD"], capture_output=True, text=True, check=True
    ).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        rows.append((parts[0][0], parts[-1]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="origin/claude/implement-hybrid-search-WEN8O")
    args = ap.parse_args()

    rows = changed_files(args.base)
    poc_dirs = sorted({p.split("/")[1] for _, p in rows if p.startswith("hf-space/") and p.count("/") >= 2})
    new_dirs = sorted({p.split("/")[1] for s, p in rows if s == "A" and p.startswith("hf-space/") and p.count("/") >= 2})
    flagged = []
    for status, path in rows:
        if path in ALWAYS_OK or path.startswith(OK_PREFIXES):
            continue
        if path.startswith("hf-space/") and path.count("/") >= 2:
            folder = path.split("/")[1]
            if folder in new_dirs and len(new_dirs) == 1:
                continue
        flagged.append(f"{status} {path}")

    print("### Scope check")
    print(f"- Files changed: {len(rows)}")
    print(f"- New POC folders: {', '.join(new_dirs) or 'none'}")
    if len(new_dirs) > 1:
        print(f"- More than one new POC folder: {', '.join(new_dirs)}")
    touched_existing = sorted(set(poc_dirs) - set(new_dirs))
    if touched_existing:
        print(f"- Existing POC folders modified: {', '.join(touched_existing)}")
    if flagged:
        print("- Outside POC scope (needs a reason in the PR body):")
        for f in flagged:
            print(f"  - `{f}`")
        return 1
    print("- All changes within POC scope.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
