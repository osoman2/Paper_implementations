# Vector Ridge Lab · Paper implementations

Working proofs of concept of recent AI research, reproduced within days of publication and
published as live demos. Research first, production always.

- **Live demos:** [Hugging Face Space `osoman2/Gradio_pocs`](https://huggingface.co/spaces/osoman2/Gradio_pocs)
  (open one directly with `?poc=POC-050`)
- **Catalog, live matching and integration prompts:** [vectorridgelabs.com/lab](https://vectorridgelabs.com/en/lab)
- **Publishing branch:** `deploy` (everything below describes that branch)

## How the Lab is built

An author agent reads AI newsletters every morning, scores each paper for AI engineering relevance
and, when one scores 8/10 or more, implements it as a new tab of the Space. Nothing it writes
reaches production without passing two independent gates.

```
Author agent (daily, Claude)       reads newsletters, scores papers (INBOX.md), writes the POC
   │  opens a PR: poc/<slug>-<date>  (or inbox/<date> on days without a POC)
   ▼
smoke (GitHub Actions)             imports the app, renders it, serves / and /config, runs every
   │                               event handler with default and example inputs, checks pins
   │                               and the website contract                    → tests/
review (GitHub Actions, Groq)      a reviewer that never saw the authoring checks the diff
   │                               against REVIEW_CHECKLIST.md; reviewer code and checklist
   │                               come from `deploy`, so a PR cannot change its own rules
   ▼
auto-merge                         only when both checks pass; `deploy` is protected, no bypass
   ▼
Deploy (GitHub Actions)            smoke again, upload hf-space/, wait until the Space is RUNNING
   ▼
vectorridgelabs.com Lab sync       rebuilds the catalog from app.py + INBOX.md
```

Weekly numbers for this loop (PRs, reviewer failures, CI blocks, failed deploys) come from
`tools/agent_metrics.py` and are kept in `METRICS.md`.

## Repository layout

| Path | What it is |
|---|---|
| `hf-space/` | The Gradio app published to the Space. One folder per POC, one `poc_tab(...)` per POC in `app.py`. |
| `INBOX.md` | Every paper scanned, with score, summary, original source link and implementation status. |
| `tests/` | Smoke suite and its harness (`smoke_harness.py` exercises every wired handler). |
| `REVIEW_CHECKLIST.md` | The rules the reviewer applies, each seeded from a real incident. |
| `tools/kb.py` | Lab knowledge base: hybrid retrieval (BM25 + trigram, RRF) over INBOX.md, app.py and the POC folders. |
| `tools/groq_review.py`, `tools/check_pr_scope.py` | The reviewer and its deterministic scope check. |
| `tools/agent_metrics.py` | Weekly metrics of the agent loop from the GitHub API. |
| `CLAUDE.md`, `docs/daily-author-prompt.md` | Instructions for the author agent (versioned like code). |
| `hybrid-search/`, `cad-score/` | The first two POCs as standalone projects, before the Space existed. |

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r hf-space/requirements.txt pytest requests
cd hf-space && python app.py          # the Space, on http://127.0.0.1:7860
cd .. && pytest tests/ -q             # the same gate CI runs
```

Explore the Lab without opening the big files:

```bash
python tools/kb.py brief                       # tabs, regions, recent scans, backlog
python tools/kb.py search "agent memory cost"  # related POCs and papers
python tools/kb.py seen "<paper title>"        # already scored or implemented?
```

## Contributing a POC

Open a PR to `deploy` from a `poc/<slug>-<date>` branch. Read `REVIEW_CHECKLIST.md` first: the new
tab is declared with `with poc_tab("Name: short claim"):`, starts with a `gr.Markdown` intro, keeps
the pinned `gradio==4.44.0`, does no heavy work at import time, and links the original paper.

## License

Code is released under the [MIT License](LICENSE). Each POC reimplements ideas from the paper or
announcement it links; credit for the research belongs to its authors.
