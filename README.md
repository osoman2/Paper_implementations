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

## POC Index

| Folder | Paper / Technique | Key Gain | Status |
|--------|-------------------|----------|--------|
| [`hybrid-search`](./hybrid-search/) | Query-Adaptive Hybrid Search (DAT + vstash) | +2–17% NDCG@10 over fixed-weight hybrid | ✅ Done |
