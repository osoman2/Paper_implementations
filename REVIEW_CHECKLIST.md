# Reviewer checklist for agent-authored POC PRs

Used by the reviewer agent, which runs in a separate context and never sees how the code was
written. Every item is pass/fail with a `file:line` reference. Any fail means "request changes".
Items marked (CI) are also enforced by `tests/test_smoke.py`; the reviewer still reads for them,
because CI only exercises default inputs.

Each item comes from a real failure. Add a new item whenever something slips through.

## Correctness
- [ ] Every name used in new code is imported or defined. (CI) — 2026-10-18: `typing.Any` missing, NameError at import.
- [ ] Data passed to Gradio components matches their schema: `gr.Dataframe` rows are lists with `headers`, not dicts. (CI) — 2026-10-18: CLM tab.
- [ ] Handlers do not crash on empty input: `None`/`""` from a cleared textbox returns a friendly message.
- [ ] No model downloads, network calls, or heavy compute at import time; load lazily inside handlers.
- [ ] Numbers shown in the UI are labelled as the paper's reported figures or as simulated, never presented as measured by the demo.

## Conventions
- [ ] New tab declared with `with poc_tab("Name: short claim"):`, appended after the last POC tab, no emojis. (CI)
- [ ] The tab's first element is a `gr.Markdown` intro: `## Title` + 2–4 sentence summary.
- [ ] New code lives in `hf-space/<paper_slug>/` plus its tab block in `app.py` and its `INBOX.md` entry. Nothing else changes.
- [ ] `hf-space/requirements.txt` locked pins unchanged (gradio==4.44.0, huggingface_hub<1.0, starlette<1.0). (CI) New packages only if the POC needs them, with a one-line reason in the PR.

## Provenance and safety
- [ ] `INBOX.md` entry has a `**Link:**` to the original source (arXiv, journal, authors' repo or blog), opened and checked. Never a newsletter, tracking, redirect, or mailto URL. If unverifiable, the line is omitted.
- [ ] No secrets, tokens, API keys, or personal email addresses in code, INBOX.md, or the PR body.
- [ ] Score in INBOX.md matches the scoring guide and the paper is genuinely >= 8 for AI engineering relevance.

## Verdict format
```
VERDICT: APPROVE | REQUEST_CHANGES
- [pass|fail] <item> — <file:line> — <note or exact fix>
```
