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

## What's Implemented

### `src/retriever.py`
Two retrievers over an in-memory corpus of 20 AI/RAG documents:

- **`SparseRetriever`** — BM25 Okapi via `rank-bm25`. Tokenises, builds inverted index, scores each document.
- **`DenseRetriever`** — `all-MiniLM-L6-v2` sentence transformer. Encodes corpus once at startup, then runs cosine similarity via dot product on L2-normalised vectors.

### `src/query_analyzer.py`
**IDF-based alpha computation** (vstash approach):

1. Builds document-frequency table over the corpus at init time.
2. For each query token, computes smoothed IDF: `log((N+1)/(df+1)) + 1`
3. Takes the mean IDF of query tokens → `mean_idf`
4. Normalises to [0,1], then applies sigmoid with steepened slope: `sigmoid((normalised - 0.5) × 8)`
5. Maps sigmoid output to `[min_alpha=0.1, max_alpha=0.9]`

Result: rare/technical terms → alpha near 0.9 (heavy sparse); common words → alpha near 0.1 (heavy dense).

### `src/fusion.py`
Two fusion strategies, both alpha-aware:

- **Score fusion** — min-max normalises each ranked list independently, then blends: `alpha × sparse_score + (1-alpha) × dense_score`
- **RRF fusion** — Reciprocal Rank Fusion with alpha-weighted contribution per list: `alpha × 1/(k+rank)` for sparse, `(1-alpha) × 1/(k+rank)` for dense. Rank-only, no score normalisation needed.

### `src/app.py`
Gradio demo with:
- Query input + example queries
- Toggle between auto (IDF-adaptive) and manual alpha
- Score fusion vs RRF selection
- Results table, alpha analysis panel, fixed-vs-adaptive diff view

## Architecture

```
Query
  │
  ├─► query_analyzer.py  ──► adaptive alpha (IDF-based sigmoid)
  │
  ├─► retriever.py (BM25)  ──► sparse_results [(doc_id, score), ...]
  │
  └─► retriever.py (Dense) ──► dense_results  [(doc_id, score), ...]
            │
            └─► fusion.py  ──► final_ranked_results
```

## Key Design Choices

- **No LLM call at query time** — alpha is computed in microseconds from corpus statistics built at init.
- **Corpus-relative IDF** — alpha adapts to the domain of your index, not a global vocabulary.
- **Sigmoid steepness = 8** — makes the transition sharp: queries are decisively sparse-heavy or dense-heavy, not perpetually balanced.
- **Alpha range [0.1, 0.9]** — never goes fully to one retriever; always blends both signals.

## Extending to Production

To replace the in-memory corpus with a real index:

```python
# retriever.py — swap SparseRetriever for Elasticsearch BM25:
from elasticsearch import Elasticsearch
# or swap DenseRetriever for Qdrant:
from qdrant_client import QdrantClient

# query_analyzer.py — build IDF from your actual corpus at index time
# and persist it; load at query time. The compute_alpha() call is unchanged.
```

The `QueryAnalyzer` only needs the corpus texts to build its IDF table. Swap the in-memory list for any iterable of document strings.

## Comparing Fixed vs Adaptive Alpha

The Gradio UI shows a live diff between `alpha=0.5` (fixed) and the adaptive alpha for every query. Try:

| Query | Expected alpha | Why |
|-------|---------------|-----|
| `BM25 Okapi` | ~0.85 | Rare technical term, sparse wins |
| `Karpukhin dense passage` | ~0.80 | Named entity, sparse wins |
| `how do transformers work` | ~0.15 | Common words, dense wins |
| `retrieval augmented generation knowledge` | ~0.35 | Mixed, leans dense |
