# Query-Adaptive Hybrid Search

## Paper

| Field | Detail |
|-------|--------|
| **Primary paper** | DAT: Dynamic Alpha Tuning for Hybrid Retrieval in RAG — arXiv:2503.23013 |
| **Related** | vstash: Local-First Hybrid Retrieval — arXiv:2604.15484 |
| **Related** | QuARI: Query-Adaptive Retrieval — arXiv:2505.21647 |

## Core Idea

Standard hybrid search (BM25 + dense vector) uses a **fixed alpha** weight across all queries, e.g. `score = α·sparse + (1-α)·dense`. This is suboptimal: keyword-heavy queries (named entities, product IDs, rare terms) benefit from higher sparse weight; paraphrased or conceptual queries need more dense weight.

**DAT** dynamically computes alpha per query by scoring top-1 results from each retriever using an LLM judge, then normalising those scores. **vstash** uses a lighter approach: it measures query IDF — rare/technical queries get higher sparse weight via sigmoid normalisation, common-word queries get higher dense weight. No LLM call needed.

This POC implements the **vstash IDF-adaptive approach** (production-friendly, no extra LLM overhead) with an optional LLM-judge mode.

## Relevance for AI Engineering

- Directly improves RAG retrieval quality with zero change to the document index
- Eliminates the tedious per-domain alpha hyperparameter tuning
- No index rebuild — drop-in replacement for fixed-weight hybrid search
- Works with any BM25 + vector store combination (Elasticsearch, Qdrant, pgvector, etc.)
- **+2–7.5 pp Precision@1 / MRR@20** (DAT) and **+17.6% relative NDCG@10** (vstash)

## How to Run

```bash
cd hybrid-search
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/app.py
```

Then open `http://localhost:7860`.

## Architecture

```
Query
  │
  ├─► query_analyzer.py  ──► adaptive alpha (IDF-based)
  │
  ├─► retriever.py (BM25)  ──► sparse_results
  │
  └─► retriever.py (Dense) ──► dense_results
            │
            └─► fusion.py  ──► final_ranked_results
```
