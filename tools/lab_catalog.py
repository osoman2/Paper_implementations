"""
Read-only parsers for the two files the rest of the system depends on: INBOX.md and hf-space/app.py.

The patterns mirror osoman2/MyWebPage `frontend_landing/scripts/build-lab-catalog.py`, which builds
the Lab on vectorridgelabs.com from this branch. Tests use them as a contract (if these parse, the
website parses), and tools/kb.py uses them to index the Lab without an LLM reading whole files.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE_DIR = ROOT / "hf-space"
APP_PY = SPACE_DIR / "app.py"
INBOX_MD = ROOT / "INBOX.md"

# Same regexes as the website catalog builder. Do not loosen them here without changing it there.
WEB_TAB_RE = re.compile(
    r'^        with (?:gr\.Tab|poc_tab)\("([^"]+)"\):[^\n]*\n            gr\.Markdown\(\n((?:\s+(?:f?"[^\n]*"|#[^\n]*)\n)+)',
    re.M,
)
WEB_IMPL_RE = re.compile(
    r"Implemented:\*\* Yes — `hf-space/([a-z0-9_]+)/` · Tab (\d+)[^(]*\((\d{4}-\d{2}-\d{2})\)"
)
WEB_HEAD_RE = re.compile(r"### \S+ \[(.+?)\]\s*—\s*Score (\d+)(?:-\d+)?/10")
POC_TAB_RE = re.compile(r'^\s*with poc_tab\("([^"]+)"\)', re.M)

# The first POCs predate the INBOX "Implemented:" convention; the website hardcodes their position.
LEGACY_TABS = {
    1: "hybrid_search", 2: "cad_score", 3: "dssw", 4: "synergistic_engine",
    5: "attribution_crisis", 6: "olr_waa", 7: "bi_stage_rag", 8: "opinion_maximization",
    9: "topomas", 10: "label_scarcity", 11: "extreme_event_ai", 22: "matryoshka_lm",
}

_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF]")


@dataclass
class Tab:
    number: int  # 1-based, declaration order == POC-NNN
    label: str
    intro: str  # the gr.Markdown intro text, joined
    line: int  # line of the `with poc_tab(` statement

    @property
    def code(self) -> str:
        return f"POC-{self.number:03d}"


@dataclass
class Paper:
    title: str
    score: int
    run_date: str  # date of the "New Papers (this run — ...)" section it appeared in
    summary: str
    source: str = ""
    link: str = ""
    poc_idea: str = ""
    implemented_folder: str = ""
    implemented_tab: int = 0
    implemented_date: str = ""
    line: int = 0
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("**", "")).strip()


def parse_tabs(src: str | None = None) -> list[Tab]:
    src = src if src is not None else APP_PY.read_text(encoding="utf-8")
    intros = {}
    for m in WEB_TAB_RE.finditer(src):
        intros[m.start()] = "".join(re.findall(r'f?"((?:[^"\\]|\\.)*)"', m.group(2)))
    tabs = []
    for i, m in enumerate(POC_TAB_RE.finditer(src), start=1):
        line_start = src.rfind("\n", 0, m.start()) + 1
        intro = intros.get(line_start, "").replace("\\n", "\n")
        tabs.append(Tab(i, m.group(1), intro, src.count("\n", 0, m.start()) + 1))
    return tabs


def web_tab_matches(src: str | None = None) -> list[str]:
    """Labels the website's regex can read. Must equal the poc_tab labels, in order."""
    src = src if src is not None else APP_PY.read_text(encoding="utf-8")
    return [m.group(1) for m in WEB_TAB_RE.finditer(src)]


def parse_inbox(text: str | None = None) -> list[Paper]:
    text = text if text is not None else INBOX_MD.read_text(encoding="utf-8")
    papers: list[Paper] = []
    run_date = ""
    offset = 0
    for block in re.split(r"\n(?=##+ )", text):
        line = text.count("\n", 0, text.find(block, offset)) + 1
        offset = max(offset, text.find(block, offset))
        run = re.match(r"## .*?(\d{4}-\d{2}-\d{2})", block)
        if run and not block.startswith("### "):
            run_date = run.group(1)
            continue
        head = WEB_HEAD_RE.match(block)
        if not head:
            continue
        quote = " ".join(
            l[2:] for l in block.splitlines() if l.startswith("> ") and "**Source:**" not in l
        )
        source = re.search(r"\*\*Source:\*\* ([^\n]+)", block)
        idea = re.search(r"\*\*POC idea:\*\* ([^\n]+)", block)
        link = re.search(r"\*\*Link:\*\* <?(https?://[^\s>]+)", block)
        impl = WEB_IMPL_RE.search(block)
        papers.append(
            Paper(
                title=head.group(1).strip(),
                score=int(head.group(2)),
                run_date=run_date,
                summary=_clean(quote),
                source=_clean(source.group(1)) if source else "",
                link=link.group(1) if link else "",
                poc_idea=_clean(idea.group(1)) if idea else "",
                implemented_folder=impl.group(1) if impl else "",
                implemented_tab=int(impl.group(2)) if impl else 0,
                implemented_date=impl.group(3) if impl else "",
                line=line,
            )
        )
    return papers


def poc_folders() -> list[str]:
    return sorted(
        p.name
        for p in SPACE_DIR.iterdir()
        if p.is_dir() and not p.name.startswith((".", "_")) and any(p.glob("*.py"))
    )


def has_emoji(text: str) -> bool:
    return bool(_EMOJI.search(text))
