---
title: Paper POCs — AI Engineering
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Paper POCs — AI Engineering

Practical reproductions of high-relevance AI/ML research papers, focused on techniques
directly applicable to RAG, retrieval, agents, and LLM evaluation.

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
