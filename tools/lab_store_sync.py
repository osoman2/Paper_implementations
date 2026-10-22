#!/usr/bin/env python3
"""
Mirror the Lab into the retrieval store (Supabase `vrl.lab_documents`) used by vectorridgelabs.com.

Runs in GitHub Actions after every merge to `deploy` (.github/workflows/lab-store-sync.yml).
The repo stays the source of truth; the store is a searchable copy for the website's RAG:

    POCs   (every tab of the Space)              -> kind=poc,   status=implemented
    papers (scanned, no tab, score >= 5)         -> kind=paper, status=scanned

Only documents whose content changed are sent; the lab-sync edge function embeds them with
gte-small and upserts. Documents that disappeared from the repo are pruned.

    LAB_SYNC_SECRET=... python tools/lab_store_sync.py              # sync
    python tools/lab_store_sync.py --dry-run                        # build documents, print counts
    python tools/lab_store_sync.py --probe "agent memory cost"      # query lab-search (no secret)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kb  # noqa: E402

FUNCTIONS_URL = os.environ.get("LAB_FUNCTIONS_URL", "https://jlavdoweaeuscqeqcfsb.supabase.co/functions/v1")
SPACE_APP = "https://osoman2-gradio-pocs.hf.space"
# Free-tier edge functions embed ~2 documents per request before hitting their compute limit (HTTP 546).
BATCH = 2
RETRYABLE = {502, 503, 504, 546}
# Papers below this score are mostly product news; indexing them crowds relevant results out.
MIN_PAPER_SCORE = 5
UA = "vrl-lab-store-sync/1.0 (+https://github.com/osoman2/Paper_implementations)"


def _clean(text: str, limit: int) -> str:
    text = re.sub(r"\*\*|`|#+ ", "", text or "")
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _paper_id(title: str) -> str:
    key = re.sub(r"\W+", " ", title.lower()).strip()
    return "paper:" + hashlib.sha1(key.encode()).hexdigest()[:12]


def _with_hash(doc: dict) -> dict:
    payload = json.dumps({k: doc[k] for k in sorted(doc)}, ensure_ascii=False, sort_keys=True)
    doc["content_hash"] = hashlib.sha1(payload.encode()).hexdigest()[:16]
    return doc


def build_documents() -> list[dict]:
    kb.build()
    docs: list[dict] = []
    for p in kb._load(kb.POCS):
        title = p["title"] if len(p["title"]) > len(p["label"]) else p["label"]
        summary = _clean(p["summary"], 1500)
        parts = [title, p["label"] if p["label"] != title else "", p["paper_title"], summary, p["doc"]]
        body = _clean(". ".join(x for x in parts if x), 4000)
        docs.append(_with_hash({
            "id": f"poc:{p['code']}",
            "kind": "poc",
            "code": p["code"],
            "title": _clean(title, 300),
            "summary": summary,
            "body": body,
            "score": p["score"] or None,
            "status": "implemented",
            "region": p["region"],
            "url": f"{SPACE_APP}/?poc={p['code']}",
            "source_url": p["link"] or None,
            "run_date": p["implemented"] or None,
        }))
    seen = set()
    for r in kb._load(kb.PAPERS):
        if r.get("implemented_folder"):
            continue  # represented by its POC
        if (r.get("score") or 0) < MIN_PAPER_SCORE:
            continue
        pid = _paper_id(r["title"])
        if pid in seen:
            continue
        seen.add(pid)
        summary = _clean(r["summary"], 1500)
        docs.append(_with_hash({
            "id": pid,
            "kind": "paper",
            "code": None,
            "title": _clean(r["title"], 300),
            "summary": summary,
            "body": _clean(f"{r['title']}. {summary} {r.get('poc_idea', '')}", 4000),
            "score": r["score"],
            "status": "scanned",
            "region": None,
            "url": r.get("link") or None,
            "source_url": r.get("link") or None,
            "run_date": r.get("run_date") or None,
        }))
    return docs


def _post(path: str, payload: dict, secret: str | None = None, timeout: int = 120, retries: int = 3):
    headers = {"Content-Type": "application/json", "User-Agent": UA}
    if secret:
        headers["x-lab-sync-secret"] = secret
    data = json.dumps(payload).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(f"{FUNCTIONS_URL}/{path}", data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read()[:300].decode("utf-8", "replace")
            if e.code in RETRYABLE and attempt < retries:
                time.sleep(3 * (attempt + 1))
                continue
            raise RuntimeError(f"{path} HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
                continue
            raise RuntimeError(f"{path} unreachable: {e.reason}") from e
    raise RuntimeError(f"{path}: retries exhausted")


def sync(secret: str) -> dict:
    docs = build_documents()
    existing = {h["id"]: h for h in _post("lab-sync", {"action": "hashes"}, secret)}
    changed = [
        d for d in docs
        if d["id"] not in existing
        or existing[d["id"]]["content_hash"] != d["content_hash"]
        or not existing[d["id"]]["has_embedding"]
    ]
    upserted, failed = 0, []
    for i in range(0, len(changed), BATCH):
        batch = changed[i : i + BATCH]
        try:
            upserted += _post("lab-sync", {"action": "upsert", "docs": batch}, secret)["upserted"]
        except RuntimeError as e:
            # Keep going: what was stored stays stored, and the next run retries only what is missing.
            failed += [d["id"] for d in batch]
            print(f"::warning::upsert failed for {[d['id'] for d in batch]}: {e}")
    pruned = _post("lab-sync", {"action": "prune", "keep_ids": [d["id"] for d in docs]}, secret)["pruned"]
    return {"documents": len(docs), "changed": len(changed), "upserted": upserted, "failed": len(failed), "pruned": pruned}


def probe(query: str, count: int = 5, kind: str = "poc") -> list[dict]:
    return _post("lab-search", {"query": query, "count": count, "kinds": [kind]})["results"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe", action="append", default=[])
    args = ap.parse_args()

    if args.dry_run:
        docs = build_documents()
        kinds = {k: sum(1 for d in docs if d["kind"] == k) for k in ("poc", "paper")}
        print(json.dumps({"documents": len(docs), **kinds, "example": docs[0]}, indent=2, ensure_ascii=False)[:1500])
        return 0
    if not args.probe:
        secret = os.environ.get("LAB_SYNC_SECRET", "").strip()
        if not secret:
            print("::notice::LAB_SYNC_SECRET not set: retrieval store not synced.")
            return 0
        result = sync(secret)
        print(f"::notice::Lab store synced: {json.dumps(result)}")
        if result["failed"]:
            print(f"::error::{result['failed']} documents could not be stored; re-run to retry them.")
            return 1
    # Same split as the website: POCs and papers are searched separately.
    for q in args.probe:
        for kind in ("poc", "paper"):
            hits = probe(q, kind=kind)
            lines = [f"{h.get('code') or ''} {h['title'][:60]} (sim {h['similarity']:.2f})" for h in hits]
            print(f"::notice title={kind} probe: {q[:50]}::" + " | ".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
