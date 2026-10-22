# Paper Implementations & POCs

Practical reproductions of AI/ML research papers — focused on techniques directly applicable to production AI engineering (RAG, retrieval, agents, LLMs).

Each POC lives in its own folder with an independent Python environment, a paper summary, a relevance assessment, and a runnable demo.

---

## Repository Structure

```
.
├── README.md                  ← this file
├── hybrid-search/             ← Query-Adaptive Hybrid Search (DAT / vstash)
│   ├── README.md              ← paper summary + relevance assessment
│   ├── requirements.txt
│   └── src/
│       ├── app.py             ← Gradio demo (entry point)
│       ├── retriever.py       ← BM25 + dense retriever
│       ├── query_analyzer.py  ← IDF-based adaptive alpha
│       └── fusion.py          ← RRF + score normalization
└── <next-poc>/
    └── ...
```

---

## Guidelines for New POCs

### Folder & naming
- Use a short kebab-case name that matches the paper's key concept: `hybrid-search`, `speculative-rag`, `colbert-retrieval`.
- Every POC needs a `README.md` that includes:
  - **Paper** — title, authors, arXiv link
  - **Core idea** — ≤5 sentences, what problem it solves and how
  - **Relevance** — why this matters for AI engineering (RAG, latency, accuracy, cost)
  - **Results** — key metric improvements from the paper
  - **How to run** — copy-paste commands

### Environment
- Each POC has its own `requirements.txt`. No monorepo-level deps.
- Python ≥ 3.10. Use `venv`:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```

### UI stack
- **Frontend only** → plain HTML/JS or a single Python script with `gradio`.
- **Backend needed** → Python + Gradio, NiceGUI, Streamlit, or Taipy.
- No full-stack boilerplate. Keep it minimal and runnable in one command.

### Code style
- Entry point is always `src/app.py`.
- No comments explaining *what* the code does — only non-obvious *why* constraints.
- No premature abstractions. Three similar lines > a helper function.

### Selecting papers to implement
Prioritise papers that are:
1. **Recent** (< 18 months) and not yet widely reproduced
2. **Directly applicable** to RAG, retrieval, agents, or LLM inference
3. **Implementable** without proprietary data or >24 GB VRAM
4. **Benchmarked** — has clear metrics to validate the implementation against

### Claude workflow commands
- `/vrl-implement` — implement a new POC from a paper or idea
- `/vrl-research` — research a paper and produce relevance assessment

---

---

## Agent delivery loop

Most code here is written by a scheduled Claude Code agent. Nothing it writes reaches the
public Space without passing three gates:

1. **Author agent** (daily scheduled task): scans paper newsletters, implements a POC on a
   `poc/<slug>-<date>` branch, runs `pytest tests/test_smoke.py` until green, and opens a PR
   into the deploy branch. It never pushes POC code to the deploy branch directly.
2. **Reviewer agent**: a separate subagent with a fresh context, which never saw the code being
   written, reviews the diff against [`REVIEW_CHECKLIST.md`](./REVIEW_CHECKLIST.md) and posts
   `APPROVE` or `REQUEST_CHANGES` with file:line findings. The author fixes and re-requests.
3. **CI smoke test** (`.github/workflows/smoke-test.yml`, required check): boots the full app,
   serves HTTP, and runs every default-input handler through Gradio's real pre/postprocess.

Merge happens only with an approving review and a green smoke test. The deploy workflow then
publishes to Hugging Face and fails unless the Space reaches `RUNNING`.

The checklist and tests grow from real incidents: the first full deploy (2026-10-18) failed on a
missing import, dict rows passed to `gr.Dataframe`, and dependency drift. The smoke test
reproduces all three.

Run locally (where huggingface.co is unreachable, stub the embedding model):

```bash
pip install -r hf-space/requirements.txt pytest httpx
SMOKE_FAKE_EMBEDDINGS=1 pytest tests/test_smoke.py -v
```

---

## POC Index

| Folder | Paper / Technique | Key Gain | Status |
|--------|-------------------|----------|--------|
| [`hybrid-search`](./hybrid-search/) | Query-Adaptive Hybrid Search (DAT + vstash) | +2–17% NDCG@10 over fixed-weight hybrid | ✅ Done |
| [`cad-score`](./cad-score/) | CAD Score: Context-Aware Semantic Diversity (Kim et al. 2025) | Objective LLM creativity evaluation without human judges | ✅ Done |

## Paper Backlog (from JournalClub.io)

Papers sourced and assessed for relevance. Sorted by implementation priority.

| Paper | Why relevant | Priority |
|-------|-------------|----------|
| A synergistic engine paradigm for real-time, context-aware decision-making | Event-driven pattern matching + state tracking — applicable to agent orchestration | Medium |
| Drop-in density-aware partitioning for tree-based anomaly detection | Improved Isolation Forest — useful for LLM/system output monitoring | Medium |
| The attribution crisis in LLM search results | Research on LLM attribution gap — informs RAG citation design | Low (research) |
