"""
Smoke test for the Vector Ridge Lab Space (hf-space/).

Runs in CI on every pull request to the deploy branch, and locally before an agent opens a PR:

    pip install -r hf-space/requirements.txt pytest requests
    pytest tests/ -q

Every check here maps to a real incident or to a contract another system depends on.
"""

from __future__ import annotations

import os
import re
import socket

import pytest

import lab_catalog as catalog
from smoke_harness import exercise

# Rules for new tabs apply from this POC number on (older tabs keep their emoji labels).
NEW_TAB_RULES_FROM = 52

LOCKED_PINS = {
    "gradio": "==4.44.0",  # the whole app is written against 4.44
    "huggingface_hub": "<1.0",  # gradio 4.44 imports HfFolder, removed in 1.x
    "starlette": "<1.0",  # starlette 1.x breaks gradio 4.44 TemplateResponse
}

NEWSLETTER_HOSTS = re.compile(
    r"journalclub\.io|alphasignal|substack\.com/redirect|beehiiv|mailchi\.mp|list-manage\.com|"
    r"links\.|click\.|/track/|utm_|mailto:",
    re.I,
)
SECRET_PATTERNS = re.compile(
    r"sk-ant-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{32,}|hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|"
    r"AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)


# ---------------------------------------------------------------------------
# Static checks (no import needed)
# ---------------------------------------------------------------------------

def _requirements() -> dict[str, str]:
    reqs = {}
    for raw in (catalog.SPACE_DIR / "requirements.txt").read_text().splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        m = re.match(r"([A-Za-z0-9_.\-\[\]]+)\s*(.*)", line)
        reqs[m.group(1).lower().replace("-", "_")] = m.group(2).replace(" ", "")
    return reqs


def test_requirements_keep_locked_pins():
    reqs = _requirements()
    for pkg, spec in LOCKED_PINS.items():
        assert pkg in reqs, f"{pkg} must stay pinned in hf-space/requirements.txt"
        assert spec in reqs[pkg].split(","), (
            f"{pkg} pin changed to '{reqs[pkg]}', expected '{spec}'. "
            "These pins keep gradio 4.44 working on the Space; change them only with a migration."
        )


def test_tab_labels_readable_by_website():
    """vectorridgelabs.com reads app.py with a strict regex; every tab must match it, in order."""
    tabs = catalog.parse_tabs()
    web = catalog.web_tab_matches()
    assert [t.label for t in tabs] == web, (
        "Some poc_tab blocks are invisible to the website catalog. Each tab must be declared at "
        "8-space indent as `with poc_tab(\"...\"):` followed directly by `gr.Markdown(` at 12 spaces "
        "with string-literal lines. Missing: "
        + ", ".join(t.label for t in tabs if t.label not in web)
    )


def test_every_tab_maps_to_one_folder():
    tabs = catalog.parse_tabs()
    implemented = {}
    for paper in catalog.parse_inbox():
        if paper.implemented_folder:
            assert paper.implemented_tab not in implemented, (
                f"Tab {paper.implemented_tab} claimed twice in INBOX.md "
                f"({implemented.get(paper.implemented_tab)} and {paper.implemented_folder})"
            )
            implemented[paper.implemented_tab] = paper.implemented_folder
    mapping = {**catalog.LEGACY_TABS, **implemented}
    folders = set(catalog.poc_folders())
    missing = [n for n in range(1, len(tabs) + 1) if n not in mapping]
    assert not missing, (
        f"Tabs {missing} have no `**Implemented:** Yes — `hf-space/<folder>/` · Tab N ...` line in "
        "INBOX.md, so the website cannot place them."
    )
    beyond = sorted(n for n in mapping if n > len(tabs))
    assert not beyond, f"INBOX.md references tabs {beyond} but app.py has only {len(tabs)}"
    ghost = sorted(f for f in mapping.values() if f not in folders)
    assert not ghost, f"INBOX.md / legacy map points to missing folders: {ghost}"
    orphan = sorted(folders - set(mapping.values()))
    assert not orphan, f"POC folders without a tab: {orphan}"


def test_new_tabs_follow_conventions():
    for tab in catalog.parse_tabs():
        if tab.number < NEW_TAB_RULES_FROM:
            continue
        assert not catalog.has_emoji(tab.label), f"{tab.code} label has an emoji: {tab.label!r}"
        assert tab.intro.lstrip().startswith("## "), f"{tab.code} intro must start with a '## ' title"


def test_inbox_links_point_to_original_sources():
    for paper in catalog.parse_inbox():
        if not paper.link:
            continue
        assert paper.link.startswith("https://"), f"Link must be https: {paper.link}"
        assert not NEWSLETTER_HOSTS.search(paper.link), (
            f"Link for '{paper.title}' looks like a newsletter or tracking URL: {paper.link}. "
            "Use the arXiv page, repo or official announcement."
        )


def test_no_secrets_committed():
    hits = []
    targets = [catalog.INBOX_MD, *catalog.SPACE_DIR.rglob("*.py"), *catalog.ROOT.glob("tools/*.py")]
    for path in targets:
        for n, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if SECRET_PATTERNS.search(line):
                hits.append(f"{path.relative_to(catalog.ROOT)}:{n}")
    assert not hits, f"Possible secrets: {hits}"


# ---------------------------------------------------------------------------
# Runtime checks
# ---------------------------------------------------------------------------

def test_app_imports(app_module):
    assert hasattr(app_module, "demo"), "app.py must define `demo` (the gr.Blocks)"


def test_tab_registry_consistent(app_module):
    import lab_nav

    assert lab_nav._counter == len(lab_nav.INDEX) == len(catalog.parse_tabs()), (
        "poc_tab() calls at runtime differ from the tabs found in app.py source"
    )


def test_config_renders(app_module):
    # Builds the page config: every component's initial value goes through postprocess here.
    config = app_module.demo.get_config_file()
    assert config["components"], "empty Blocks config"


def test_every_event_handler_runs(app_module):
    report = exercise(app_module.demo, timeout_s=int(os.environ.get("SMOKE_HANDLER_TIMEOUT", "90")))
    print(report.summary())
    assert report.results, "no event handlers found"
    assert not report.failures, report.summary()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_server_serves_pages(app_module):
    requests = pytest.importorskip("requests")
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
    port = _free_port()
    demo = app_module.demo
    demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True)
    try:
        session = requests.Session()
        session.trust_env = False
        for path in ("/", "/config"):
            r = session.get(f"http://127.0.0.1:{port}{path}", timeout=60)
            assert r.status_code == 200, f"GET {path} -> {r.status_code}: {r.text[:300]}"
    finally:
        demo.close()
