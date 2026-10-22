# Prompt of the "JournalClub Daily Scanner" scheduled task (author agent)

Versioned here so changes to the agent's instructions go through review like code.
The repo's CLAUDE.md is loaded automatically and holds the details; this prompt is the run order.
Paste everything below the line into the scheduled task.

---

Repository: osoman2/Paper_implementations. Its CLAUDE.md is the agent guide; follow it.

0. Check out branch deploy and pull. Install test deps if missing:
   pip install -q -r hf-space/requirements.txt pytest requests
1. Run `python tools/kb.py brief`. Do NOT read INBOX.md or hf-space/app.py in full on this run;
   use tools/kb.py and read only the line ranges it points to.
2. Fix before you add: list open agent PRs
   `gh api "repos/osoman2/Paper_implementations/pulls?state=open&base=deploy" -q '.[] | select(.head.ref|startswith("poc/")) | "\(.number) \(.head.ref)"'`
   For each, read its failing checks (`gh api repos/osoman2/Paper_implementations/commits/<sha>/check-runs`)
   and the "Agent review: FAIL" comment. Check out the branch, apply the fixes, run
   `SMOKE_ALLOW_OFFLINE=1 pytest tests/ -q` until green, push. Do not open a new POC while one is open.
3. Scan new emails with the Gmail connector from the senders used in tools/daily_scan.py
   (JournalClub, AlphaSignal), received after the "last scan" date in the brief.
   For every candidate paper run `python tools/kb.py seen "<title>"`. Exit 0 means already
   scored or implemented: list it as re-listed with the existing score, do not re-score it.
4. Score the new ones (1-10, scoring guide in .claude/skills/journal-scan.md). Write today's
   section to run.md in the existing INBOX format (first line "Last updated: ...", then
   "## New Papers (this run — YYYY-MM-DD)" and one "### <emoji> [Title] — Score N/10" entry each),
   then `python tools/kb.py add-run run.md`.
5. `python tools/kb.py next --since <today>`. If it returns a paper scored >= 8 that is worth a demo:
   - Get the ORIGINAL source URL from the email (arXiv, repo, official blog). Never the newsletter,
     a tracking link or mailto; omit the line if unsure.
   - Read REVIEW_CHECKLIST.md. Use `python tools/kb.py template` for the tab structure and the
     insertion line, and `python tools/kb.py search "<topic>"` to name overlapping tabs.
   - Follow "Shipping a POC" in CLAUDE.md: branch poc/<slug>-<YYYY-MM-DD>, implement
     hf-space/<slug>/, add the tab and the INBOX `**Implemented:**` + `**Link:**` lines,
     smoke test green, open the PR with REST, enable auto-merge with REST. If enabling
     auto-merge fails, leave the PR open and say so in the email: Osman merges it by hand.
   - PR body: paper link, score, what each sub-tab shows, smoke test summary line.
   - Wait for checks: poll `gh api repos/osoman2/Paper_implementations/pulls/<n>` until `merged`
     is true or a check fails (max 25 min). If it fails, fix once (step 2 procedure); otherwise
     leave it for tomorrow's run.
   - Only after the PR is merged: `gh api repos/osoman2/MyWebPage/dispatches -f event_type=lab-poc-added`.
     If that fails, do not retry (MyWebPage's daily action at 10:30 Lima opens the same PR).
   - Never set commit statuses, never post a review verdict, never push code to deploy.
6. No new POC today: commit INBOX.md directly to the deploy branch and push. On Mondays also run
   `python tools/agent_metrics.py --write` and include METRICS.md in that commit.
7. Send Osman a confirmation and an email in Spanish: papers scanned, scores and why, re-listed
   items, PR link and check status if a POC was opened (Space and Lab update on merge).
