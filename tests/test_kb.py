"""The Lab knowledge base (tools/kb.py) must stay consistent with INBOX.md and app.py."""

import shutil

import pytest

import kb
import lab_catalog as catalog


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Point kb at a copy of INBOX.md and a temp index dir, so tests never touch the repo."""
    inbox = tmp_path / "INBOX.md"
    shutil.copy(catalog.INBOX_MD, inbox)
    monkeypatch.setattr(catalog, "INBOX_MD", inbox)
    monkeypatch.setattr(kb, "KB_DIR", tmp_path / "kb")
    monkeypatch.setattr(kb, "PAPERS", tmp_path / "kb" / "papers.jsonl")
    monkeypatch.setattr(kb, "POCS", tmp_path / "kb" / "pocs.jsonl")
    monkeypatch.setattr(kb, "MANIFEST", tmp_path / "kb" / "manifest.json")
    return inbox


def test_build_indexes_every_tab(sandbox):
    manifest = kb.build()
    pocs = kb._load(kb.POCS)
    assert manifest["pocs"] == len(catalog.parse_tabs()) == len(pocs)
    assert all(p["folder"] for p in pocs), "every POC record needs its hf-space folder"
    assert [p["tab"] for p in pocs] == list(range(1, len(pocs) + 1))


def test_seen_finds_implemented_paper(sandbox):
    paper = next(p for p in catalog.parse_inbox() if p.implemented_folder)
    hits = kb.seen(paper.title)
    assert hits[0]["similarity"] >= kb.SEEN_THRESHOLD


def test_seen_rejects_unrelated_title(sandbox):
    assert kb.seen("Quantum error correction on superconducting qubits")[0]["similarity"] < kb.SEEN_THRESHOLD


def test_backlog_excludes_implemented(sandbox):
    implemented = {p.title for p in catalog.parse_inbox() if p.implemented_folder}
    assert not implemented & {r["title"] for r in kb.backlog()}


def test_search_returns_relevant_poc(sandbox):
    hits = kb.search("hybrid search BM25 dense fusion", k=5, scope="pocs")
    assert any(h["folder"] == "hybrid_search" for h in hits)


def test_template_points_to_next_tab(sandbox):
    t = kb.template()
    assert t["next_tab"] == len(catalog.parse_tabs()) + 1
    assert t["insert_before"].startswith("wire_navigation(")
    assert t["last_tab_block"].startswith("        with poc_tab(")


def test_add_run_inserts_section_at_top(sandbox):
    section = (
        "Last updated: 2099-01-01 · test run\n"
        "## New Papers (this run — 2099-01-01)\n\n"
        "### ⚪ [A test paper about retrieval] — Score 5/10\n"
        "> Test summary.\n"
        "> **Source:** Test · 2099-01-01\n"
    )
    kb.add_run(section)
    text = sandbox.read_text(encoding="utf-8")
    assert text.splitlines()[1].startswith("Last updated: 2099-01-01")
    first_run = next(l for l in text.splitlines() if l.startswith("## "))
    assert "2099-01-01" in first_run
    assert catalog.parse_inbox(text)[0].title == "A test paper about retrieval"
    assert not kb.is_stale()
