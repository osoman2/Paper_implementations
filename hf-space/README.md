---
title: Vector Ridge Lab — POC demos
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
---

# Paper POCs — AI Engineering

Practical reproductions of high-relevance AI/ML research papers, focused on techniques
directly applicable to RAG, retrieval, agents, and LLM evaluation.

Source, tests and the agent review loop: [osoman2/Paper_implementations](https://github.com/osoman2/Paper_implementations)
(branch `deploy`). Every change reaches this Space only after a smoke test and an independent review.
Catalog and integration prompts: [vectorridgelabs.com/lab](https://vectorridgelabs.com/en/lab).

## Tabs

### 🔍 Hybrid Search — Score 10/10
**Papers:** DAT (arXiv:2503.23013) + vstash (arXiv:2604.15484)

Replaces fixed alpha in hybrid search with a per-query IDF-adaptive weight.
High-IDF (rare/technical) queries lean sparse (BM25); low-IDF (conceptual) queries lean dense.
+2–17% NDCG@10 over fixed-weight hybrid baseline.

### 📊 CAD Score — Score 9/10
**Paper:** Kim et al. 2025 — PeerJ Computer Science

Context-Aware Diversity metric for LLM output evaluation.
Unlike raw embedding spread, CAD penalises responses that merely paraphrase the prompt.

### 📈 DSSW — Score 9/10
**Paper:** Dynamic Sub-Sequence Warping: A Representation-Based Similarity Measure for Long Time Series

Segments each series into overlapping sub-sequences, then runs a DTW-style DP alignment
at the sub-sequence level. Useful for comparing embedding-drift trajectories, agent-state
histories, or sequential retrieval-score runs. O(n/stride · m/stride) vs O(n·m) for DTW.

### 🌐 Opinion Maximization with Fairness — Score 8/10
**Paper:** Opinion Maximization Based on Fairness in Social Networks

Greedy influence maximisation with a fairness budget that guarantees minority-group seed coverage.
Demonstrates that requiring ≥1 minority seed often increases total spread (the minority cluster
is dense internally) while closing the majority/minority coverage gap.
RAG analogy: graph-RAG seed selection, knowledge graph traversal fairness, agent activation routing.
