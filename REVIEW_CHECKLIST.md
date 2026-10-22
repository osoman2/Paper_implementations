# Review checklist for agent-written POC pull requests

Used by the reviewer agent (`.github/workflows/agent-review.yml`), which did not write the code.
Each item is seeded from a real incident or from a contract another system depends on. The reviewer
answers every item with PASS, FAIL or N/A and a `file:line` reference. Any FAIL blocks the merge.

Items marked **(CI)** are also enforced deterministically by `tests/test_smoke.py`; the reviewer still
reads them, because a test can pass for the wrong reason.

## Correctness

1. **Every name used is defined or imported.** New code uses no name that is not imported or defined
   in scope. Incident 2026-10-18: `typing.Any` used in `app.py` without the import, the Space crashed
   at start. **(CI)**
2. **Gradio components receive the shape they expect.** `gr.Dataframe` gets a list of row lists (or a
   pandas DataFrame), never a list of dicts. `gr.Plot` gets a figure, `gr.Markdown` a string, a handler
   with N outputs returns N values. **(CI, for every wired handler)**
3. **Handlers degrade on empty input.** An empty textbox or missing API key returns a readable message,
   not an exception. **(CI)**
4. **The demo shows what the paper claims.** Numbers, formulas and the tab intro match the source.
   Simulations and toy versions are labelled as such in the UI ("Illustration only", "simulated").

## Safety of the Space

5. **No heavy work at import time.** No model download, network call or large computation at module
   import; load on first use. The app must import in under 120 s. **(CI)**
6. **Pins unchanged.** `hf-space/requirements.txt` keeps `gradio==4.44.0`, `huggingface_hub<1.0`,
   `starlette<1.0`. Any new package is justified in the PR body and is light enough for the free CPU
   Space. **(CI for the pins)**
7. **No secrets or personal data.** No API keys, tokens, personal emails or phone numbers in code,
   INBOX.md or the PR body. **(CI for common key formats)**

## Contracts with the Lab and the website

8. **Tab declaration.** New tab declared as `with poc_tab("Name: short claim"):` at 8-space indent,
   appended after the last POC tab and before `wire_navigation(...)`, no emojis in the label, first
   element a `gr.Markdown(` intro with a `## ` title and a 2-4 sentence summary. The website parses this
   with a strict regex. **(CI)**
9. **INBOX entry.** The paper's INBOX.md entry has
   `**Implemented:** Yes — \`hf-space/<folder>/\` · Tab N in \`hf-space/app.py\` (YYYY-MM-DD)` with the
   correct N (= the new tab's position). **(CI)**
10. **Original source link.** `**Link:** <https://...>` points to the arXiv page, the repo or the authors'
    official announcement. Never a newsletter, tracking redirect or mailto. If no reliable original
    exists, the line is omitted, not guessed. **(CI for known newsletter hosts)**

## Scope

11. **The diff touches only what a POC needs:** the new `hf-space/<folder>/`, its tab block in
    `hf-space/app.py`, and its INBOX.md entry. Anything else (workflows,
    tests, other POCs, requirements) needs a reason in the PR body. **(scope check step)**
12. **No unrelated refactors** of existing tabs, even if they look like improvements.

## Verdict

PASS only if every item is PASS or N/A. Otherwise FAIL, with the exact fix for each failing item.
