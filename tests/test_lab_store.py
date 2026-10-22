"""Documents mirrored to the website's retrieval store (no network)."""

import lab_catalog as catalog
import lab_store_sync as store


def test_documents_cover_every_tab_and_unimplemented_paper():
    docs = store.build_documents()
    ids = [d["id"] for d in docs]
    assert len(ids) == len(set(ids)), "document ids must be unique"
    pocs = [d for d in docs if d["kind"] == "poc"]
    assert len(pocs) == len(catalog.parse_tabs())
    assert all(d["url"].endswith(f"?poc={d['code']}") for d in pocs)
    implemented = {p.title for p in catalog.parse_inbox() if p.implemented_folder}
    papers = [d for d in docs if d["kind"] == "paper"]
    assert not implemented & {d["title"] for d in papers}
    assert all(d["score"] >= store.MIN_PAPER_SCORE for d in papers)


def test_documents_fit_the_store_schema():
    for d in store.build_documents():
        assert d["kind"] in ("poc", "paper") and d["status"] in ("implemented", "scanned")
        assert d["title"] and len(d["title"]) <= 400
        assert d["body"] and len(d["body"]) <= 8000 and len(d["summary"]) <= 4000
        assert d["score"] is None or 0 <= d["score"] <= 10
        assert len(d["content_hash"]) == 16


def test_hash_changes_with_content():
    a = store._with_hash({"id": "x", "body": "one"})["content_hash"]
    b = store._with_hash({"id": "x", "body": "two"})["content_hash"]
    assert a != b
