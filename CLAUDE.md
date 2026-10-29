# Paper_implementations — agent guide

Deploy branch: `deploy`. `hf-space/` is published to the Hugging Face
Space `osoman2/Gradio_pocs` and read by vectorridgelabs.com (Lab catalog) from this branch.

## Read the Lab through the knowledge base, not the raw files

`INBOX.md` (~200 KB) and `hf-space/app.py` (~490 KB) are too large to read on every run. Use:

```bash
python tools/kb.py brief                 # state of the Lab: tabs, regions, recent runs, backlog
python tools/kb.py seen "<paper title>"  # exit 0 = already scored or implemented: skip it
python tools/kb.py next                  # best paper >= 8 without a tab
python tools/kb.py search "<topic>" -k 5 # related papers/POCs (for "overlaps Tab N" notes)
python tools/kb.py template              # last tab block + exact insertion line for a new tab
python tools/kb.py add-run run.md        # prepend today's INBOX section and re-index
python tools/source_feeds.py show --since <date>  # Hugging Face Daily Papers + new GitHub repos
```

Sources: newsletters by Gmail (JournalClub, AlphaSignal) and the feeds above. The `Source feeds` Action
(12:17 UTC) publishes the feeds to the `feeds` branch, because agent sandboxes cannot call the GitHub
search API. Never edit or push the `feeds` branch by hand: the Action overwrites it daily.

The index (`kb/`, git-ignored) rebuilds itself when INBOX.md or app.py change. Read a slice of a big
file only when a command points you to a line (`Read` with offset/limit), never the whole file.

## Shipping a POC: always through a pull request

1. Branch `poc/<slug>-<YYYY-MM-DD>` from the deploy branch.
2. Add `hf-space/<slug>/`, one tab block at the line `kb.py template` gives, the INBOX entry with
   `**Implemented:**` and `**Link:**` lines.
3. Run the smoke test until green:
   `SMOKE_ALLOW_OFFLINE=1 pytest tests/ -q` (offline mode only tolerates model-hub network errors).
4. Open a PR to the deploy branch and enable auto-merge. Agent sandboxes have no GitHub GraphQL,
   so `gh pr create` / `gh pr merge --auto` fail; use REST:
   ```bash
   gh api repos/osoman2/Paper_implementations/pulls -f title="POC: ..." -f head=poc/<slug>-<date> \
     -f base=deploy -F body=@pr.md -q .number
   gh api -X PUT repos/osoman2/Paper_implementations/pulls/<n>/ccr/auto_merge -f merge_method=squash
   ```
   It merges only when the `smoke` and `review` checks pass. If enabling auto-merge answers
   422 "clean status", the checks already passed: merge with
   `gh api -X PUT repos/osoman2/Paper_implementations/pulls/<n>/merge -f merge_method=squash`.
   Never merge a PR whose checks are failing or pending, and never push to the deploy branch:
   it is protected without bypass, so every change, including INBOX.md, goes through a PR.
5. Days without a new POC: commit INBOX.md (and METRICS.md on Mondays) on a branch
   `inbox/<YYYY-MM-DD>`, open a PR to `deploy` and enable auto-merge the same way. Only the smoke
   test runs on it (the reviewer only reviews `poc/*` branches).

Every rule the reviewer applies is in `REVIEW_CHECKLIST.md`; read it before writing the tab. The
reviewer runs inside GitHub Actions (`agent-review.yml`, Groq by default) and posts its verdict as a PR
comment. Never set commit statuses or comment a verdict yourself.

## Hard rules

- `hf-space/requirements.txt` pins stay: `gradio==4.44.0`, `huggingface_hub<1.0`, `starlette<1.0`.
- No model downloads or network calls at import time; load on first use.
- New tab: `with poc_tab("Name: short claim"):` at 8-space indent, no emojis, first element
  `gr.Markdown(` with a `## ` title. The website parses this exact shape.
- `gr.Dataframe` values are lists of row lists, never lists of dicts.
