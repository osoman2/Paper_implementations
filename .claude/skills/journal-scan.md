# Journal Scan

Scan JournalClub.io emails for new AI engineering papers, score their relevance with LLM, and propose the next POC to implement.

## What this does

1. Runs `tools/daily_scan.py` to fetch and score any unprocessed emails from `noreply@newsletter.journalclub.io`
2. Reads `INBOX.md` for the full ranked report
3. Presents new papers sorted by relevance score (1–10)
4. Proposes the top candidate for implementation

## Steps

<execute_bash>
python tools/daily_scan.py   # from the repository root
</execute_bash>

After running the script, read INBOX.md and present a summary to the user:

- List new papers grouped by score: 🔴 High (8–10), 🟡 Medium (6–7), ⚪ Low (1–5)
- For the top-scoring paper, show the `poc_idea` field and ask: "Want me to implement this as the next POC?"
- If no new papers: "No new papers since last scan. Here's the current backlog:" then show previously tracked high-relevance papers from INBOX.md

## Scoring guide (for context)

| Score | Meaning |
|-------|---------|
| 8–10 | Direct AI engineering: RAG, hybrid search, LLM eval, embeddings, agents, inference |
| 6–7 | Applicable: NLP, recommendation, knowledge graphs, anomaly detection for ML systems |
| 4–5 | Tangential: general ML, classic CV/NLP without LLM angle |
| 1–3 | Not relevant: IoT, climate, biology, domain-specific non-AI |

## Context for relevance scoring

The target reader is an **AI engineer** building production systems:
- RAG pipelines (chunking, retrieval, reranking, evaluation)
- LLM applications (prompt engineering, output evaluation, latency, cost)
- Hybrid search (BM25 + dense, fusion strategies, adaptive weighting)
- AI agents (orchestration, tool use, memory, multi-agent)
- Embeddings and vector stores
- LLM inference optimisation (speculative decoding, quantisation, caching)
