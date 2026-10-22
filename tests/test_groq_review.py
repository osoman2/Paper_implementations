"""The reviewer's verdict logic must not be fooled by the model (no network: Groq is mocked)."""

import json
import subprocess
import sys

import groq_review


def _run(monkeypatch, tmp_path, review: dict) -> dict:
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr(groq_review, "call_groq", lambda *a, **k: review)
    monkeypatch.setattr(groq_review, "collect_diff", lambda base: "diff")
    monkeypatch.setattr(sys, "argv", ["groq_review", "--base", "HEAD", "--out", str(tmp_path)])
    assert groq_review.main() == 0
    return json.loads((tmp_path / "verdict.json").read_text())


def test_pass_with_failed_item_becomes_fail(monkeypatch, tmp_path):
    verdict = _run(monkeypatch, tmp_path, {
        "verdict": "PASS",
        "summary": "looks fine",
        "items": [{"n": 1, "result": "PASS"}, {"n": 2, "result": "FAIL", "fix": "use rows as lists"}],
    })
    assert verdict["verdict"] == "FAIL" and verdict["failed_items"] == [2]
    comment = (tmp_path / "comment.md").read_text()
    assert comment.startswith("## Agent review: FAIL") and "use rows as lists" in comment


def test_clean_review_passes(monkeypatch, tmp_path):
    items = [{"n": n, "result": "PASS" if n % 3 else "N/A"} for n in range(1, 13)]
    verdict = _run(monkeypatch, tmp_path, {"verdict": "PASS", "summary": "ok", "items": items})
    assert verdict["verdict"] == "PASS"


def test_missing_verdict_is_fail(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, {"items": []})["verdict"] == "FAIL"


def test_collect_diff_runs_on_this_repo():
    base = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert "DIFF OF MODIFIED FILES" in groq_review.collect_diff(base)
