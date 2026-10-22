#!/usr/bin/env python3
"""
Lab knowledge base: store and retrieve what the Lab knows without reading whole files.

INBOX.md (~200 KB) and hf-space/app.py (~490 KB) are ~175k tokens together. The daily agent used to
read both on every run. This tool indexes them once into small JSONL files under kb/ and answers the
questions the agent actually asks, in a few hundred tokens:

    python tools/kb.py build                 # re-index INBOX.md + app.py + POC folders -> kb/
    python tools/kb.py brief                 # compact state of the Lab (start every run with this)
    python tools/kb.py next [--since DATE]   # best paper scored >= 8 not implemented yet
    python tools/kb.py seen "paper title"    # already scored or implemented? (exit 0 = seen)
    python tools/kb.py search "query" -k 5   # hybrid search over papers and POCs
    python tools/kb.py template              # last tab block + insertion line, to add a new tab
    python tools/kb.py add-run run.md        # insert today's INBOX section at the top, re-index

Retrieval is hybrid and dependency-free (it must run in sandboxes without model downloads):
BM25 over words + character-trigram similarity, fused with Reciprocal Rank Fusion, the same idea
as POC-001 (Query-Adaptive Hybrid Search). Titles are boosted. Nothing here calls an LLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_catalog as catalog  # noqa: E402

sys.path.insert(0, str(catalog.SPACE_DIR))
import lab_regions as regions  # noqa: E402  (gradio-free, same classification as the Space)

KB_DIR = catalog.ROOT / "kb"
PAPERS = KB_DIR / "papers.jsonl"
POCS = KB_DIR / "pocs.jsonl"
MANIFEST = KB_DIR / "manifest.json"

SEEN_THRESHOLD = 0.55
_STOP = set(
    "a an and are as at be by for from has have in into is it its of on or that the this to via vs "
    "with without new how why what your you we our their can more than over".split()
)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.lower())
    return [w for w in words if w not in _STOP and len(w) > 1]


def _trigrams(text: str) -> Counter:
    t = " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "
    return Counter(t[i : i + 3] for i in range(len(t) - 2))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))


def _title_similarity(a: str, b: str) -> float:
    """Blend of trigram cosine and word Jaccard; robust to reworded newsletter titles."""
    ta, tb = set(_tokens(a)), set(_tokens(b))
    jac = len(ta & tb) / len(ta | tb) if ta and tb else 0.0
    return 0.6 * _cosine(_trigrams(a), _trigrams(b)) + 0.4 * jac


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avgdl = sum(map(len, docs)) / max(len(docs), 1)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for d, tf in zip(self.docs, self.tf):
            s = 0.0
            for t in query:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out.append(s)
        return out


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _docstring(folder: str) -> str:
    init = catalog.SPACE_DIR / folder / "__init__.py"
    files = [init] if init.exists() and init.stat().st_size > 0 else sorted((catalog.SPACE_DIR / folder).glob("*.py"))
    for f in files:
        m = re.match(r'\s*(?:#[^\n]*\n\s*)*[rub]*("""|\'\'\')(.*?)\1', f.read_text(encoding="utf-8", errors="ignore"), re.S)
        if m:
            return re.sub(r"\s+", " ", m.group(2)).strip()[:600]
    return ""


def build() -> dict:
    tabs = catalog.parse_tabs()
    papers = catalog.parse_inbox()
    folder_by_tab = dict(catalog.LEGACY_TABS)
    paper_by_folder = {}
    for p in papers:
        if p.implemented_folder:
            folder_by_tab[p.implemented_tab] = p.implemented_folder
            paper_by_folder[p.implemented_folder] = p

    pocs = []
    for t in tabs:
        folder = folder_by_tab.get(t.number, "")
        p = paper_by_folder.get(folder)
        intro_title = re.search(r"##\s*([^\n]+)", t.intro)
        score = re.search(r"Score (\d+)/10", t.intro)
        pocs.append(
            {
                "code": t.code,
                "tab": t.number,
                "label": catalog._EMOJI.sub("", t.label).strip(),
                "title": intro_title.group(1).strip() if intro_title else t.label,
                "folder": folder,
                "region": regions.region_for(t.code, t.label),
                "score": int(score.group(1)) if score else (p.score if p else 0),
                "paper_title": p.title if p else "",
                "link": p.link if p else "",
                "implemented": p.implemented_date if p else "",
                "summary": (p.summary if p else "") or re.sub(r"\s+", " ", t.intro)[:600],
                "doc": _docstring(folder) if folder else "",
                "app_line": t.line,
            }
        )

    # One record per distinct paper title (the same paper is often re-listed in later runs).
    merged: dict[str, dict] = {}
    for p in papers:
        key = re.sub(r"\W+", " ", p.title.lower()).strip()
        rec = merged.get(key)
        d = p.to_dict()
        d.pop("tags", None)
        if rec is None:
            d["mentions"] = 1
            merged[key] = d
        else:
            rec["mentions"] += 1
            rec["score"] = max(rec["score"], d["score"])
            for k in ("link", "poc_idea", "implemented_folder", "implemented_date", "source"):
                if not rec.get(k) and d.get(k):
                    rec[k] = d[k]
            if d["implemented_tab"]:
                rec["implemented_tab"] = d["implemented_tab"]
            rec["first_seen"] = min(rec.get("first_seen") or rec["run_date"], d["run_date"] or rec["run_date"])
    for rec in merged.values():
        rec.setdefault("first_seen", rec["run_date"])

    KB_DIR.mkdir(exist_ok=True)
    _write_jsonl(PAPERS, sorted(merged.values(), key=lambda r: (r["run_date"], r["score"]), reverse=True))
    _write_jsonl(POCS, pocs)
    manifest = {
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inbox_sha1": hashlib.sha1(catalog.INBOX_MD.read_bytes()).hexdigest()[:12],
        "app_sha1": hashlib.sha1(catalog.APP_PY.read_bytes()).hexdigest()[:12],
        "papers": len(merged),
        "paper_mentions": len(papers),
        "pocs": len(pocs),
        "last_run": max((p.run_date for p in papers if p.run_date), default=""),
        "source_bytes": catalog.INBOX_MD.stat().st_size + catalog.APP_PY.stat().st_size,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _load(path: Path) -> list[dict]:
    if not path.exists():
        build()
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def is_stale() -> bool:
    if not MANIFEST.exists():
        return True
    m = json.loads(MANIFEST.read_text())
    return (
        m.get("inbox_sha1") != hashlib.sha1(catalog.INBOX_MD.read_bytes()).hexdigest()[:12]
        or m.get("app_sha1") != hashlib.sha1(catalog.APP_PY.read_bytes()).hexdigest()[:12]
    )


def _ensure_fresh() -> None:
    if is_stale():
        build()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _paper_text(r: dict) -> str:
    return f"{r['title']} {r['title']} {r['summary']} {r.get('poc_idea', '')}"


def _poc_text(r: dict) -> str:
    return f"{r['title']} {r['label']} {r['paper_title']} {r['summary']} {r['doc']}"


def search(query: str, k: int = 5, scope: str = "all") -> list[dict]:
    """Hybrid retrieval: BM25 + trigram similarity, fused with RRF (k=60)."""
    _ensure_fresh()
    items: list[tuple[str, dict, str]] = []
    if scope in ("all", "papers"):
        items += [("paper", r, _paper_text(r)) for r in _load(PAPERS)]
    if scope in ("all", "pocs"):
        items += [("poc", r, _poc_text(r)) for r in _load(POCS)]
    if not items:
        return []
    bm = BM25([_tokens(t) for _, _, t in items]).scores(_tokens(query))
    qg = _trigrams(query)
    tri = [_cosine(qg, _trigrams(r.get("title", "") + " " + r.get("label", ""))) for _, r, _ in items]
    fused = Counter()
    for scores in (bm, tri):
        order = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
        for rank, i in enumerate(order[:100]):
            if scores[i] > 0:
                fused[i] += 1.0 / (60 + rank + 1)
    out = []
    for i, s in fused.most_common(k):
        kind, r, _ = items[i]
        out.append({"kind": kind, "rrf": round(s, 4), **r})
    return out


def seen(title: str, k: int = 3) -> list[dict]:
    _ensure_fresh()
    pool = [("paper", r, r["title"]) for r in _load(PAPERS)] + [
        ("poc", r, f"{r['title']} {r['paper_title']}") for r in _load(POCS)
    ]
    scored = sorted(
        ((_title_similarity(title, text), kind, r) for kind, r, text in pool), key=lambda x: x[0], reverse=True
    )
    return [{"similarity": round(s, 3), "kind": kind, **r} for s, kind, r in scored[:k]]


def backlog(min_score: int = 8, since: str = "") -> list[dict]:
    """Papers scored >= min_score with no tab, minus those matching an implemented POC."""
    _ensure_fresh()
    pocs = _load(POCS)
    implemented_titles = [f"{p['title']} {p['paper_title']}" for p in pocs]
    out = []
    for r in _load(PAPERS):
        if r["score"] < min_score or r.get("implemented_folder"):
            continue
        if since and (r.get("run_date") or "") < since:
            continue
        best = max((_title_similarity(r["title"], t) for t in implemented_titles), default=0.0)
        if best >= SEEN_THRESHOLD:
            continue
        out.append({**r, "closest_poc_similarity": round(best, 3)})
    return sorted(out, key=lambda r: (r["score"], r["run_date"]), reverse=True)


# ---------------------------------------------------------------------------
# Writing (store)
# ---------------------------------------------------------------------------

def add_run(section_md: str) -> dict:
    """Insert a '## New Papers (this run — DATE)' section at the top of INBOX.md and re-index.

    Also refreshes the 'Last updated:' line from the section's first line if it starts with it.
    """
    text = catalog.INBOX_MD.read_text(encoding="utf-8")
    section = section_md.strip("\n") + "\n\n"
    last_updated = None
    if section.startswith("Last updated:"):
        last_updated, _, section = section.partition("\n")
        section = section.lstrip("\n")
    if not section.startswith("## "):
        raise SystemExit("run section must start with '## New Papers (this run — YYYY-MM-DD)'")
    first = text.find("\n## ")
    head, rest = text[: first + 1], text[first + 1 :]
    if last_updated:
        head = re.sub(r"^Last updated:.*$", last_updated.strip(), head, count=1, flags=re.M)
    catalog.INBOX_MD.write_text(head + "\n" + section + rest.lstrip("\n"), encoding="utf-8")
    return build()


def template() -> dict:
    src = catalog.APP_PY.read_text(encoding="utf-8")
    lines = src.splitlines()
    tabs = catalog.parse_tabs(src)
    last = tabs[-1]
    end = next(i for i, l in enumerate(lines) if l.strip().startswith("wire_navigation("))
    block = "\n".join(lines[last.line - 1 : end]).rstrip()
    return {
        "next_tab": last.number + 1,
        "next_code": f"POC-{last.number + 1:03d}",
        "insert_before_line": end + 1,
        "insert_before": lines[end].strip(),
        "last_tab_block": block,
        "inbox_implemented_line": (
            f"**Implemented:** Yes — `hf-space/<folder>/` · Tab {last.number + 1} in `hf-space/app.py` (YYYY-MM-DD)"
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt_paper(r: dict) -> str:
    impl = f" · implemented Tab {r['implemented_tab']} ({r['implemented_folder']})" if r.get("implemented_folder") else ""
    link = f" · {r['link']}" if r.get("link") else ""
    return f"[{r['score']}/10] {r['title']} · {r['run_date']}{impl}{link}"


def _fmt_poc(r: dict) -> str:
    return f"{r['code']} [{r['region']}] {r['label']} · hf-space/{r['folder']}/"


def cmd_brief(_args) -> None:
    _ensure_fresh()
    m = json.loads(MANIFEST.read_text())
    pocs = _load(POCS)
    papers = _load(PAPERS)
    regions = Counter(p["region"] for p in pocs)
    recent_runs = sorted({p["run_date"] for p in papers if p["run_date"]}, reverse=True)[:3]
    bl = backlog()
    print(f"# Lab brief (index built {m['built_at']}, last scan {m['last_run']})")
    print(f"- {len(pocs)} POC tabs; next tab: {template()['next_code']}")
    print("- Regions: " + ", ".join(f"{k} {v}" for k, v in regions.most_common()))
    print(f"- {len(papers)} distinct papers scored ({m['paper_mentions']} mentions)")
    print("- Latest POCs:")
    for p in pocs[-5:]:
        print(f"  - {_fmt_poc(p)}")
    for d in recent_runs:
        rows = sorted((p for p in papers if p["run_date"] == d), key=lambda r: -r["score"])[:4]
        print(f"- Run {d}: " + "; ".join(f"{r['score']} {r['title'][:60]}" for r in rows))
    print(f"- Backlog (>= 8, not implemented): {len(bl)}")
    for r in bl[:5]:
        print(f"  - {_fmt_paper(r)}")


def cmd_next(args) -> None:
    bl = backlog(args.min_score, args.since)
    if not bl:
        print("NONE")
        sys.exit(1)
    r = bl[0]
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return
    print(_fmt_paper(r))
    print(f"summary: {r['summary']}")
    if r.get("poc_idea"):
        print(f"poc_idea: {r['poc_idea']}")
    print(f"INBOX.md line ~{r['line']} · closest existing POC similarity {r['closest_poc_similarity']}")


def cmd_seen(args) -> None:
    hits = seen(args.title, args.k)
    for h in hits:
        what = _fmt_poc(h) if h["kind"] == "poc" else _fmt_paper(h)
        print(f"{h['similarity']:.2f}  {h['kind']:5}  {what}")
    sys.exit(0 if hits and hits[0]["similarity"] >= args.threshold else 1)


def cmd_search(args) -> None:
    for h in search(args.query, args.k, args.scope):
        what = _fmt_poc(h) if h["kind"] == "poc" else _fmt_paper(h)
        print(f"{h['rrf']:.4f}  {h['kind']:5}  {what}")
        if args.verbose:
            print(f"        {(h.get('summary') or '')[:300]}")


def cmd_build(_args) -> None:
    m = build()
    print(json.dumps(m, indent=2))


def cmd_template(args) -> None:
    t = template()
    if args.json:
        print(json.dumps(t, ensure_ascii=False, indent=2))
        return
    print(f"# Next tab: {t['next_code']} (Tab {t['next_tab']})")
    print(f"# Insert the new block before line {t['insert_before_line']}: {t['insert_before']}")
    print(f"# INBOX line: {t['inbox_implemented_line']}")
    print("# Last tab block (copy its structure, 8-space indent, gr.Markdown intro first):")
    print(t["last_tab_block"])


def cmd_add_run(args) -> None:
    md = Path(args.file).read_text(encoding="utf-8") if args.file != "-" else sys.stdin.read()
    print(json.dumps(add_run(md), indent=2))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    sub.add_parser("brief").set_defaults(fn=cmd_brief)
    p = sub.add_parser("next")
    p.add_argument("--min-score", type=int, default=8)
    p.add_argument("--since", default="", help="only papers from runs on/after YYYY-MM-DD")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_next)
    p = sub.add_parser("seen")
    p.add_argument("title")
    p.add_argument("-k", type=int, default=3)
    p.add_argument("--threshold", type=float, default=SEEN_THRESHOLD)
    p.set_defaults(fn=cmd_seen)
    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--scope", choices=["all", "papers", "pocs"], default="all")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_search)
    p = sub.add_parser("template")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_template)
    p = sub.add_parser("add-run")
    p.add_argument("file", help="markdown file with the run section, or - for stdin")
    p.set_defaults(fn=cmd_add_run)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
