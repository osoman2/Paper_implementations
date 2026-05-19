"""
Paper POCs — multi-tab Gradio demo.

Tabs:
  1. Query-Adaptive Hybrid Search (DAT / vstash) — score 10/10
  2. CAD Score: Context-Aware Semantic Diversity — score 9/10
  3. Dynamic Sub-Sequence Warping (DSSW) — score 9/10
  4. Synergistic Engine (CEP + Declarative Rules for RAG) — score 8/10
  5. Attribution Crisis (LLM Search Attribution Scorer) — score 8/10
  6. OLR-WAA: Adaptive Drift-Resilient Online Regression — score 8/10
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Tab 1 — Query-Adaptive Hybrid Search
# ---------------------------------------------------------------------------
from hybrid_search.retriever import Document, SparseRetriever, DenseRetriever
from hybrid_search.query_analyzer import QueryAnalyzer
from hybrid_search.fusion import score_fusion, rrf_fusion

CORPUS = [
    Document("d1",  "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", {}),
    Document("d2",  "Attention Is All You Need — the Transformer architecture paper by Vaswani et al.", {}),
    Document("d3",  "GPT-4 Technical Report — OpenAI's large multimodal language model", {}),
    Document("d4",  "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al.)", {}),
    Document("d5",  "Dense Passage Retrieval for Open-Domain Question Answering (Karpukhin et al.)", {}),
    Document("d6",  "BM25 Okapi probabilistic ranking model for information retrieval", {}),
    Document("d7",  "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction", {}),
    Document("d8",  "LlamaIndex is a data framework for building LLM-powered retrieval applications", {}),
    Document("d9",  "LangChain provides tools to compose language model chains and agents", {}),
    Document("d10", "Vector databases like Qdrant, Pinecone, and Weaviate store embeddings at scale", {}),
    Document("d11", "Elasticsearch supports full-text search using inverted index and BM25 scoring", {}),
    Document("d12", "Fine-tuning sentence transformers improves domain-specific semantic similarity", {}),
    Document("d13", "Reciprocal Rank Fusion combines ranked lists from heterogeneous retrieval systems", {}),
    Document("d14", "Hybrid search blends sparse BM25 and dense vector retrieval for better recall", {}),
    Document("d15", "Cross-encoder reranking rescores candidate documents with a BERT-based model", {}),
    Document("d16", "Mixtral 8x7B — sparse mixture of experts model for efficient language generation", {}),
    Document("d17", "Anthropic Claude 3 — constitutional AI approach to safe and helpful assistants", {}),
    Document("d18", "Tree of Thoughts prompting enables LLMs to explore multiple reasoning paths", {}),
    Document("d19", "Self-RAG: Learning to retrieve, generate, and critique with reflection tokens", {}),
    Document("d20", "Multi-vector retrieval with ColBERT outperforms single-vector bi-encoder models", {}),
]

corpus_texts = [doc.text for doc in CORPUS]
doc_map = {doc.id: doc for doc in CORPUS}

print("Loading retrieval models (first run downloads ~90 MB)…")
sparse_ret = SparseRetriever(CORPUS)
dense_ret  = DenseRetriever(CORPUS)
analyzer   = QueryAnalyzer(corpus_texts)
print("Retrieval models ready.")


def hs_search(query: str, fusion_method: str, top_k: int, force_alpha: float, auto_alpha: bool):
    if not query.strip():
        return "", "", ""

    alpha = analyzer.compute_alpha(query) if auto_alpha else round(force_alpha, 2)
    q_type = analyzer.query_type(alpha)

    sparse_res = sparse_ret.search(query, top_k=20)
    dense_res  = dense_ret.search(query, top_k=20)

    if fusion_method == "Score fusion":
        results = score_fusion(sparse_res, dense_res, alpha, top_k=top_k)
    else:
        results = rrf_fusion(sparse_res, dense_res, alpha, top_k=top_k)

    rows = []
    for rank, (doc_id, score) in enumerate(results, 1):
        doc = doc_map[doc_id]
        rows.append(f"| {rank} | {doc_id} | {score:.4f} | {doc.text} |")

    table = (
        "| Rank | ID | Score | Document |\n"
        "|------|----|-------|----------|\n"
        + "\n".join(rows)
    )

    alpha_info = (
        f"**Alpha (sparse weight): {alpha}**  \n"
        f"Query type: *{q_type}*  \n\n"
        f"{'🔡 High IDF → sparse (BM25) preferred' if alpha >= 0.65 else '🔍 Low IDF → dense (semantic) preferred' if alpha <= 0.35 else '⚖️ Balanced — both retrievers contribute equally'}"
    )

    fixed_alpha = 0.5
    fixed_results = score_fusion(sparse_res, dense_res, fixed_alpha, top_k=top_k)
    adaptive_ids = [r[0] for r in results]
    fixed_ids    = [r[0] for r in fixed_results]
    diff_count   = sum(1 for i in adaptive_ids if i not in fixed_ids)
    compare_info = (
        f"Fixed alpha=0.5 top-{top_k}: `{', '.join(fixed_ids)}`  \n"
        f"Adaptive alpha={alpha} top-{top_k}: `{', '.join(adaptive_ids)}`  \n\n"
        f"**{diff_count} document(s) differ** between fixed and adaptive ranking."
    )

    return table, alpha_info, compare_info


# ---------------------------------------------------------------------------
# Tab 2 — CAD Score
# ---------------------------------------------------------------------------
from cad_score.cad_score import all_metrics
from cad_score.llm_client import generate_responses, generate_responses_high_temp

HAS_API_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))

EXAMPLE_PROMPTS = [
    "Give me 3 different ways to explain recursion to a beginner.",
    "Write an opening line for a mystery novel.",
    "What are the benefits of exercise?",
    "Describe the colour blue without using the word blue.",
    "Propose a startup idea in the AI space.",
]

EXAMPLE_MANUAL = """\
The quick brown fox jumps over the lazy dog.
A swift auburn fox leaps across a sleepy hound.
The nimble red fox vaults over the resting dog.
A fast tawny fox bounds over the tired canine.
The agile fox hops above the dozing dog."""


def _fmt_pairwise(distances: list[float]) -> str:
    if not distances:
        return ""
    return "\n".join(f"Pair {i+1}: {d:.4f}" for i, d in enumerate(distances))


def _build_cad_outputs(m: dict, responses: list[str]):
    scores_md = f"""
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **CAD score** | **{m['cad']}** | Higher = more context-aware diversity |
| Raw embedding spread | {m['raw_spread']} | Mean pairwise cosine distance |
| Context penalty | {m['context_penalty']} | 1 − mean(sim(response, prompt)) |
| Mean prompt similarity | {m['mean_prompt_similarity']} | How much responses echo the prompt |
| Distinct-2 | {m['distinct_2']} | Lexical diversity (unique bigrams ratio) |
| Embedding spread | {m['embedding_spread']} | Spread without context adjustment |
| Self-BLEU-2 | {m['self_bleu_2']} | Lexical overlap (lower = more diverse) |
""".strip()

    pair_md = "**Pairwise cosine distances between responses:**\n\n" + _fmt_pairwise(m["pairwise_distances"])

    explain_md = f"""
**CAD = raw_spread × context_penalty**

`{m['raw_spread']} × {m['context_penalty']} = {m['cad']}`

**Why context penalty matters:**
If all responses are highly similar to the prompt (`mean_prompt_sim = {m['mean_prompt_similarity']}`),
they're probably just paraphrasing the question rather than exploring different ideas.
CAD discounts this by multiplying with `1 − prompt_sim`.
""".strip()

    resp_text = "\n\n---\n\n".join(
        f"**Response {i+1}** (prompt_sim={m['prompt_similarities'][i]}):\n{r}"
        for i, r in enumerate(responses)
    )

    return scores_md, pair_md, explain_md, resp_text


def cad_evaluate_manual(prompt: str, raw_responses: str):
    responses = [r.strip() for r in raw_responses.strip().split("\n") if r.strip()]
    if len(responses) < 2:
        return "Need at least 2 responses (one per line).", "", "", ""
    try:
        m = all_metrics(prompt, responses)
    except Exception as e:
        return str(e), "", "", ""
    return _build_cad_outputs(m, responses)


def cad_evaluate_live(prompt: str, n_responses: int, use_creative_system: bool, progress=gr.Progress()):
    if not HAS_API_KEY:
        return "Set ANTHROPIC_API_KEY environment variable to use live generation.", "", "", ""
    if not prompt.strip():
        return "Enter a prompt first.", "", "", ""

    progress(0.1, desc="Generating responses…")
    try:
        fn = generate_responses_high_temp if use_creative_system else generate_responses
        responses = fn(prompt, n=n_responses)
    except Exception as e:
        return f"API error: {e}", "", "", ""

    progress(0.7, desc="Computing CAD score…")
    try:
        m = all_metrics(prompt, responses)
    except Exception as e:
        return str(e), "", "", ""

    progress(1.0)
    return _build_cad_outputs(m, responses)


# ---------------------------------------------------------------------------
# Tab 3 — Dynamic Sub-Sequence Warping (DSSW)
# ---------------------------------------------------------------------------
from dssw.algorithm import dssw_distance, dtw_distance, euclidean_distance

# ---------------------------------------------------------------------------
# Tab 4 — Synergistic Engine (CEP + Declarative Rules)
# ---------------------------------------------------------------------------
from synergistic_engine import DecisionEngine, Event, EventType

# ---------------------------------------------------------------------------
# Tab 5 — Attribution Crisis
# ---------------------------------------------------------------------------
from attribution_crisis import AttributionScorer, Document as AttrDoc, RagResponse, AttributionReport

# ---------------------------------------------------------------------------
# Tab 6 — OLR-WAA: Adaptive Online Regression
# ---------------------------------------------------------------------------
from olr_waa import OLRWAARegressor, DriftDetector, StreamSimulator


def olr_run(
    n_steps: int,
    drift_type: str,
    drift_point: int,
    noise_std: float,
    a_before: float,
    a_after: float,
    n_experts: int,
    eta: float,
    forgetting_factor: float,
):
    """Run OLR-WAA simulation and return metrics + plots."""
    sim = StreamSimulator(
        n_steps=n_steps,
        drift_type=drift_type,
        drift_point=int(drift_point),
        noise_std=noise_std,
        a_before=a_before,
        a_after=a_after,
    )
    X, Y = sim.generate()

    model = OLRWAARegressor(
        n_features=1,
        n_experts=n_experts,
        eta=eta,
        forgetting_factor=forgetting_factor,
    )
    detector = DriftDetector(threshold=40.0)

    # Static baseline: simple exponential smoothing (no adaptation)
    baseline_preds = []
    ema = float(Y[0])
    alpha_ema = 0.1
    for y in Y:
        baseline_preds.append(ema)
        ema = alpha_ema * float(y) + (1 - alpha_ema) * ema

    records = []
    for t, (x, y) in enumerate(zip(X, Y)):
        rec = model.step(x, float(y))
        drift_now = detector.update(rec["abs_error"], t)
        rec["drift_detected"] = drift_now
        records.append(rec)

    steps      = list(range(n_steps))
    y_true     = Y.tolist()
    y_pred_olr = [r["y_pred"] for r in records]
    abs_err_olr = [r["abs_error"] for r in records]
    abs_err_base = [abs(float(y) - p) for y, p in zip(Y, baseline_preds)]

    # Rolling MAE (window=20)
    w = 20
    roll_mae_olr  = [np.mean(abs_err_olr[max(0, i-w):i+1])  for i in steps]
    roll_mae_base = [np.mean(abs_err_base[max(0, i-w):i+1]) for i in steps]

    weight_history = np.array([r["weights"] for r in records])

    # ── Plots ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.patch.set_facecolor("#f9f9f9")

    # Plot 1: Predictions vs true
    ax = axes[0]
    ax.plot(steps, y_true,     color="#555", linewidth=1.0, label="True", alpha=0.7)
    ax.plot(steps, y_pred_olr, color="#4CAF50", linewidth=1.5, label="OLR-WAA")
    ax.plot(steps, baseline_preds, color="#FF9800", linewidth=1.2,
            linestyle="--", label="EMA baseline")
    if drift_type != "linear":
        ax.axvline(drift_point, color="red", linewidth=1.2, linestyle=":", label="Drift point")
    for dp in detector.drift_points:
        ax.axvline(dp, color="#9C27B0", linewidth=0.8, linestyle="-.", alpha=0.7)
    ax.set_title("Predictions vs Ground Truth", fontsize=11)
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Rolling MAE comparison
    ax = axes[1]
    ax.plot(steps, roll_mae_olr,  color="#4CAF50", linewidth=1.8, label="OLR-WAA")
    ax.plot(steps, roll_mae_base, color="#FF9800", linewidth=1.5,
            linestyle="--", label="EMA baseline")
    if drift_type != "linear":
        ax.axvline(drift_point, color="red", linewidth=1.2, linestyle=":",
                   label="Drift point")
    ax.set_title(f"Rolling MAE (window={w})", fontsize=11)
    ax.set_xlabel("Step")
    ax.set_ylabel("MAE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Expert weight evolution
    ax = axes[2]
    cmap = plt.get_cmap("tab10")
    for i in range(n_experts):
        ax.plot(steps, weight_history[:, i], linewidth=1.2,
                color=cmap(i), label=f"Expert {i+1}", alpha=0.8)
    if drift_type != "linear":
        ax.axvline(drift_point, color="red", linewidth=1.2, linestyle=":",
                   label="Drift point")
    ax.set_title("Expert Weight Evolution (WAA)", fontsize=11)
    ax.set_xlabel("Step")
    ax.set_ylabel("Normalised weight")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # ── Summary metrics ────────────────────────────────────────────────────
    mae_olr  = np.mean(abs_err_olr)
    mae_base = np.mean(abs_err_base)
    improvement = (mae_base - mae_olr) / mae_base * 100 if mae_base > 0 else 0

    post_drift_olr  = np.mean(abs_err_olr[drift_point:])  if drift_type != "linear" else mae_olr
    post_drift_base = np.mean(abs_err_base[drift_point:]) if drift_type != "linear" else mae_base
    post_improvement = (post_drift_base - post_drift_olr) / post_drift_base * 100 if post_drift_base > 0 else 0

    n_detected = len(detector.drift_points)

    summary = f"""
| Metric | OLR-WAA | EMA Baseline | Δ |
|--------|---------|--------------|---|
| Overall MAE | **{mae_olr:.3f}** | {mae_base:.3f} | {improvement:+.1f}% |
| Post-drift MAE | **{post_drift_olr:.3f}** | {post_drift_base:.3f} | {post_improvement:+.1f}% |
| Drift events detected | {n_detected} | — | — |
| Steps to adapt (approx.) | ~{max(5, int(10/eta))} | ∞ | — |
""".strip()

    explain = f"""
**How OLR-WAA works:**

1. **Expert pool** — {n_experts} online linear regressors trained in parallel with different learning rates.
2. **WAA weights** — each step, the weight of expert _i_ is updated:
   `log_w_i ← log_w_i × {forgetting_factor} − {eta} × (ŷ_i − y)²`
3. **Forgetting factor** ({forgetting_factor}) — discounts past losses; smaller values = faster adaptation.
4. **Ensemble prediction** = softmax(log_weights) · expert_predictions
5. **Drift detection** — Page-Hinkley test flags distribution shift at step(s): `{detector.drift_points or ['none detected']}`

**Why this matters for AI engineering:**
- Retrieval score distributions shift as knowledge bases grow.
- Embedding models change — cosine thresholds drift over time.
- OLR-WAA can track these changes **without retraining** a full model.
""".strip()

    return summary, explain, fig

AC_CORPUS = [
    AttrDoc("ac1", "Retrieval-Augmented Generation (Lewis et al., 2020)",
            "Retrieval-Augmented Generation (RAG) combines a parametric memory with a non-parametric memory. "
            "The model retrieves relevant documents from an external corpus and conditions generation on them. "
            "RAG models are trained end-to-end and can be updated by swapping the document index. "
            "The retriever uses dense passage retrieval with FAISS for efficient approximate nearest neighbour search.",
            "https://arxiv.org/abs/2005.11401"),
    AttrDoc("ac2", "Dense Passage Retrieval (Karpukhin et al., 2020)",
            "Dense Passage Retrieval (DPR) uses dual-encoder BERT models to embed questions and passages. "
            "It outperforms BM25 for open-domain QA. DPR fine-tunes encoders on question-passage pairs. "
            "Passages are pre-embedded and indexed with FAISS for millisecond retrieval at scale.",
            "https://arxiv.org/abs/2004.04906"),
    AttrDoc("ac3", "BM25 Okapi Ranking",
            "BM25 is a probabilistic bag-of-words retrieval function. It ranks documents based on "
            "term frequency and inverse document frequency. BM25 uses saturation to cap the impact of "
            "very high term frequencies. It remains a strong baseline for keyword-heavy queries.",
            "https://en.wikipedia.org/wiki/Okapi_BM25"),
    AttrDoc("ac4", "Self-RAG (Asai et al., 2023)",
            "Self-RAG learns to retrieve, generate and critique using reflection tokens. "
            "The model decides when to retrieve additional passages, evaluates retrieved passages "
            "for relevance and supports generation claims, and rates the quality of its own outputs. "
            "Self-RAG outperforms standard RAG on factual accuracy benchmarks.",
            "https://arxiv.org/abs/2310.11511"),
    AttrDoc("ac5", "RAGAS: Automated Evaluation for RAG Pipelines",
            "RAGAS evaluates RAG pipelines on faithfulness, answer relevancy and context recall. "
            "Faithfulness measures whether the answer is grounded in the retrieved context. "
            "Answer relevancy checks how well the answer addresses the question. "
            "Context recall assesses whether all necessary information was retrieved.",
            "https://arxiv.org/abs/2309.15217"),
]

_ac_scorer = AttributionScorer(AC_CORPUS)

AC_PRESET_RESPONSES = {
    "Well-attributed (cites all sources)": RagResponse(
        query="How does RAG combine retrieval and generation?",
        answer=(
            "RAG combines a parametric memory with a non-parametric memory by retrieving relevant documents "
            "from an external corpus and conditioning generation on them [ac1]. "
            "The retriever typically uses Dense Passage Retrieval (DPR) with dual-encoder BERT models "
            "and FAISS for efficient approximate nearest neighbour search [ac2]. "
            "For keyword-heavy queries, BM25 remains a strong baseline [ac3]."
        ),
        cited_ids=["ac1", "ac2", "ac3"],
    ),
    "Partially attributed (misses one source)": RagResponse(
        query="How does RAG combine retrieval and generation?",
        answer=(
            "RAG combines parametric and non-parametric memory. The model retrieves relevant documents "
            "from an external corpus and conditions generation on them [ac1]. "
            "Dense Passage Retrieval uses dual-encoder BERT models pre-embedded with FAISS. "
            "BM25 remains a strong baseline for keyword-heavy queries."
        ),
        cited_ids=["ac1"],
    ),
    "High exploitation risk (no citations)": RagResponse(
        query="How does RAG combine retrieval and generation?",
        answer=(
            "RAG combines a parametric memory with a non-parametric memory. "
            "The model retrieves relevant documents from an external corpus and conditions generation on them. "
            "RAG models are trained end-to-end and can be updated by swapping the document index. "
            "The retriever uses dense passage retrieval with FAISS for efficient approximate nearest neighbour search. "
            "Dense Passage Retrieval outperforms BM25 for open-domain QA."
        ),
        cited_ids=[],
    ),
    "Hallucinated answer (no real source)": RagResponse(
        query="What is the best LLM for production RAG?",
        answer=(
            "The best LLM for production RAG is QuantumGPT-7, which was released in 2024 by DeepMind. "
            "It achieves 98% factual accuracy on all benchmarks and requires only 2GB of VRAM. "
            "It natively supports 10 million token context windows with no performance degradation."
        ),
        cited_ids=[],
    ),
    "Self-RAG adaptive retrieval": RagResponse(
        query="How does Self-RAG improve answer quality?",
        answer=(
            "Self-RAG learns to retrieve, generate and critique using reflection tokens [ac4]. "
            "The model decides when to retrieve additional passages and evaluates their relevance. "
            "This approach outperforms standard RAG on factual accuracy benchmarks. "
            "Evaluation can be automated using frameworks like RAGAS [ac5]."
        ),
        cited_ids=["ac4", "ac5"],
    ),
}


def ac_run_preset(preset_name: str):
    response = AC_PRESET_RESPONSES[preset_name]
    report   = _ac_scorer.score(response)
    return _ac_format_report(report)


def ac_run_custom(query: str, answer: str, cited_ids_text: str):
    cited = [c.strip() for c in cited_ids_text.replace(",", " ").split() if c.strip()]
    response = RagResponse(query=query, answer=answer, cited_ids=cited)
    report   = _ac_scorer.score(response)
    return _ac_format_report(report)


def _ac_format_report(report: AttributionReport):
    risk_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[report.exploitation_risk]

    summary_md = f"""
| Metric | Value | Notes |
|--------|-------|-------|
| **Attribution score** | **{report.attribution_score:.2f}** | 0 = none, 1 = perfect |
| Exploitation risk | {risk_icon} **{report.exploitation_risk}** | HIGH = reproduced without credit |
| Citation recall (top-3) | {report.citation_recall:.0%} | Fraction of top sources cited |
| Max lexical coverage | {report.max_lexical_coverage:.1%} | Highest n-gram overlap with any source |
| Max semantic coverage | {report.max_semantic_coverage:.2f} | Highest cosine sim with any source |
| Mean semantic coverage | {report.mean_semantic_coverage:.2f} | Avg cosine sim across all sources |
""".strip()

    rows = []
    for a in report.source_attributions:
        cited_icon = "✅" if a.is_cited else "⬜"
        rows.append(
            f"| {cited_icon} | {a.doc_id} | {a.doc_title[:45]} | "
            f"{a.lexical_overlap:.1%} | {a.semantic_overlap:.2f} |"
        )
    source_table = (
        "| Cited | Doc ID | Title | Lexical overlap | Semantic sim |\n"
        "|-------|--------|-------|----------------|---------------|\n"
        + "\n".join(rows)
    )

    explain_md = f"""
**How exploitation risk is computed:**

`exploit_signal = max_lexical_overlap × (1 − citation_recall)`

`{report.max_lexical_coverage:.3f} × (1 − {report.citation_recall:.2f}) = {report.max_lexical_coverage * (1 - report.citation_recall):.3f}`

- **HIGH** (> 0.25): Large portions of a source were reproduced without citation — likely ecosystem exploitation.
- **MEDIUM** (0.10–0.25): Moderate content reuse with partial attribution.
- **LOW** (< 0.10): Either content is mostly original or sources are properly cited.

**Attribution score formula:**

`attr_score = citation_recall × 0.5 + (1 − exploit_signal) × 0.5`

`= {report.citation_recall:.2f} × 0.5 + (1 − {report.max_lexical_coverage * (1 - report.citation_recall):.3f}) × 0.5 = {report.attribution_score:.2f}`
""".strip()

    return summary_md, source_table, explain_md

_se_engine = DecisionEngine()

SE_EVENT_PRESETS = {
    "Keyword/technical query (high IDF)": {
        "type": EventType.QUERY,
        "payload": {"text": "BM25 Okapi term frequency inverse document frequency", "idf_score": 0.82},
        "source": "user",
    },
    "Semantic/conceptual query (low IDF)": {
        "type": EventType.QUERY,
        "payload": {"text": "how do language models understand context?", "idf_score": 0.22},
        "source": "user",
    },
    "Balanced hybrid query (mid IDF)": {
        "type": EventType.QUERY,
        "payload": {"text": "retrieval augmented generation pipeline", "idf_score": 0.50},
        "source": "user",
    },
    "Context retrieved from vector store": {
        "type": EventType.CONTEXT_RETRIEVED,
        "payload": {"chunks": 5, "score": 0.87},
        "source": "retriever",
    },
    "Tool call (web search)": {
        "type": EventType.TOOL_CALLED,
        "payload": {"tool": "web_search", "query": "latest RAG techniques 2025"},
        "source": "agent",
    },
    "Low-confidence response": {
        "type": EventType.RESPONSE_GENERATED,
        "payload": {"confidence": 0.38, "tokens": 120},
        "source": "llm",
    },
    "High-confidence response": {
        "type": EventType.RESPONSE_GENERATED,
        "payload": {"confidence": 0.91, "tokens": 85},
        "source": "llm",
    },
    "Error event": {
        "type": EventType.ERROR,
        "payload": {"message": "retrieval timeout", "code": 504},
        "source": "retriever",
    },
}

ACTION_COLORS = {
    "USE_SPARSE_RETRIEVAL":  "#2196F3",
    "USE_DENSE_RETRIEVAL":   "#9C27B0",
    "USE_HYBRID_SEARCH":     "#4CAF50",
    "FALLBACK_RETRIEVAL":    "#F44336",
    "RERANK_RESULTS":        "#FF9800",
    "REGENERATE_HIGH_TEMP":  "#E91E63",
    "SERVE_FROM_CACHE":      "#607D8B",
}

ACTION_LABELS = {
    "USE_SPARSE_RETRIEVAL":  "BM25 / Sparse Retrieval",
    "USE_DENSE_RETRIEVAL":   "Dense / Semantic Retrieval",
    "USE_HYBRID_SEARCH":     "Hybrid Search (α ≈ 0.5)",
    "FALLBACK_RETRIEVAL":    "Fallback Retrieval",
    "RERANK_RESULTS":        "Cross-Encoder Reranking",
    "REGENERATE_HIGH_TEMP":  "Regenerate (High Temp)",
    "SERVE_FROM_CACHE":      "Cache Hit — Skip Retrieval",
}


def se_emit_event(preset_name: str, event_log: list):
    preset = SE_EVENT_PRESETS[preset_name]
    event = Event(type=preset["type"], payload=preset["payload"], source=preset["source"])
    result = _se_engine.process_event(event)

    # Build event log entry
    action_label = ACTION_LABELS.get(result.action, result.action or "No decision")
    log_entry = {
        "event": preset_name,
        "type": event.type.value,
        "action": action_label,
        "fired_rules": result.fired_rules,
    }
    if event_log is None:
        event_log = []
    event_log = event_log + [log_entry]

    # Build rule evaluation table
    rows = []
    for name, fired, action, priority in result.all_evaluations:
        icon = "✅" if fired else "⬜"
        rows.append(f"| {icon} | {name} | P{priority} | `{action}` |")
    rule_table = (
        "| Status | Rule | Priority | Action |\n"
        "|--------|------|----------|--------|\n"
        + "\n".join(rows)
    )

    # Build stream log
    stream_lines = []
    for i, e in enumerate(_se_engine.stream.recent(), 1):
        stream_lines.append(f"**{i}.** `{e.type.value}` from `{e.source}` — {e.payload}")
    stream_md = "\n\n".join(stream_lines) if stream_lines else "_empty_"

    decision_md = f"## Decision: {action_label}\n\n{result.explanation}"

    return decision_md, rule_table, stream_md, event_log


def se_reset(event_log: list):
    _se_engine.reset()
    return "## Decision: _reset — stream cleared_", "", "_empty_", []


def se_format_log(event_log: list) -> str:
    if not event_log:
        return "_No events emitted yet._"
    rows = []
    for i, entry in enumerate(event_log, 1):
        fired = ", ".join(entry["fired_rules"]) if entry["fired_rules"] else "_none_"
        rows.append(f"| {i} | `{entry['type']}` | {entry['event'][:35]} | {entry['action']} | {fired} |")
    return (
        "| # | Type | Event | Decision | Fired Rules |\n"
        "|---|------|-------|----------|-------------|\n"
        + "\n".join(rows)
    )

DSSW_EXAMPLES = {
    "Embedding drift (shifted)": (
        [0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.7, 0.75, 0.8, 0.78, 0.7, 0.6,
         0.5, 0.42, 0.38, 0.35, 0.32, 0.3, 0.28, 0.25, 0.22, 0.2, 0.19, 0.18,
         0.17, 0.16, 0.15, 0.14, 0.13, 0.12],
        [0.08, 0.12, 0.18, 0.28, 0.42, 0.58, 0.68, 0.73, 0.79, 0.77, 0.69, 0.58,
         0.48, 0.40, 0.36, 0.33, 0.30, 0.28, 0.26, 0.23, 0.20, 0.19, 0.18, 0.17,
         0.16, 0.15, 0.14, 0.13, 0.12, 0.11],
    ),
    "Retrieval score trajectories (different speed)": (
        [0.9, 0.88, 0.85, 0.83, 0.80, 0.78, 0.75, 0.73, 0.70, 0.68,
         0.65, 0.63, 0.60, 0.58, 0.55, 0.53, 0.50, 0.48, 0.45, 0.43,
         0.40, 0.38, 0.35, 0.33, 0.30, 0.28, 0.25, 0.23, 0.20, 0.18],
        [0.9, 0.87, 0.83, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50,
         0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.18, 0.17, 0.16, 0.15,
         0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05],
    ),
    "Agent memory: similar patterns, noisy": (
        [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4,
         5, 4, 3, 2, 1, 2, 3, 4, 5, 4],
        [1.1, 2.2, 3.1, 3.9, 5.1, 3.8, 2.9, 2.1, 0.9, 2.0, 3.2, 4.1, 4.9, 3.9,
         3.1, 2.0, 1.1, 1.9, 3.1, 4.0, 5.2, 3.7, 3.0, 2.2, 0.8, 2.1, 3.0, 4.2,
         4.8, 3.8],
    ),
    "Completely unrelated series": (
        [1, 1, 1, 1, 5, 5, 5, 5, 1, 1, 1, 1, 5, 5, 5, 5, 1, 1, 1, 1,
         5, 5, 5, 5, 1, 1, 1, 1, 5, 5],
        [3, 3.1, 3.2, 3.1, 3, 2.9, 2.8, 2.9, 3, 3.1, 3.2, 3.1, 3, 2.9, 2.8, 2.9,
         3, 3.1, 3.2, 3.1, 3, 2.9, 2.8, 2.9, 3, 3.1, 3.2, 3.1, 3, 2.9],
    ),
}


def _parse_series(text: str) -> list[float]:
    text = text.strip().strip("[]")
    return [float(x.strip()) for x in text.replace(";", ",").split(",") if x.strip()]


def dssw_run(
    ts1_text: str,
    ts2_text: str,
    subseq_len: int,
    stride: int,
):
    try:
        ts1 = _parse_series(ts1_text)
        ts2 = _parse_series(ts2_text)
    except Exception:
        return "Could not parse input — use comma-separated numbers.", None

    if len(ts1) < subseq_len or len(ts2) < subseq_len:
        return (
            f"Series too short for sub-sequence length {subseq_len}. "
            f"Use at least {subseq_len} values in each series.",
            None,
        )

    try:
        dssw_dist, D, path = dssw_distance(ts1, ts2, subseq_len, stride)
        dtw_dist  = dtw_distance(ts1, ts2)
        eucl_dist = euclidean_distance(ts1, ts2)
    except Exception as e:
        return f"Computation error: {e}", None

    n_subs1 = D.shape[0]
    n_subs2 = D.shape[1]

    summary = f"""
| Metric | Value | Notes |
|--------|-------|-------|
| **DSSW distance** | **{dssw_dist:.4f}** | Normalised; lower = more similar |
| Standard DTW | {dtw_dist:.4f} | Point-level alignment |
| Euclidean (truncated) | {eucl_dist:.4f} | No time-warping |
| Sub-sequences (series 1) | {n_subs1} | len={subseq_len}, stride={stride} |
| Sub-sequences (series 2) | {n_subs2} | len={subseq_len}, stride={stride} |
| Warping path steps | {len(path)} | — |
""".strip()

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.patch.set_facecolor("#f9f9f9")

    # 1. Raw time series
    ax = axes[0]
    ax.plot(ts1, label="Series 1", color="#2196F3", linewidth=1.8)
    ax.plot(ts2, label="Series 2", color="#FF5722", linewidth=1.8, linestyle="--")
    ax.set_title("Input Series", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlabel("Time step")
    ax.grid(True, alpha=0.3)

    # 2. Sub-sequence distance matrix
    ax = axes[1]
    im = ax.imshow(D, aspect="auto", cmap="YlOrRd", origin="lower")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if len(path):
        px = [p[1] for p in path]
        py = [p[0] for p in path]
        ax.plot(px, py, "b-o", markersize=4, linewidth=1.5, label="Warping path")
        ax.legend(fontsize=8)
    ax.set_xlabel("Sub-seqs of Series 2")
    ax.set_ylabel("Sub-seqs of Series 1")
    ax.set_title("DSSW Distance Matrix + Path", fontsize=11)

    # 3. Distance bar chart comparison
    ax = axes[2]
    methods = ["DSSW\n(normalised×100)", "DTW\n(normalised×100)", "Euclidean\n(normalised×100)"]
    scale = max(dtw_dist, eucl_dist, 1e-9) / 100
    vals   = [dssw_dist * 100 / (scale if scale else 1),
              dtw_dist / (scale if scale else 1),
              eucl_dist / (scale if scale else 1)]
    colors = ["#4CAF50", "#2196F3", "#FF9800"]
    bars   = ax.bar(methods, vals, color=colors, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Distance Comparison (relative)", fontsize=11)
    ax.set_ylabel("Relative distance")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    return summary, fig


def dssw_load_example(example_name: str):
    ts1, ts2 = DSSW_EXAMPLES[example_name]
    return str(ts1)[1:-1], str(ts2)[1:-1]


# ---------------------------------------------------------------------------
# Combined multi-tab app
# ---------------------------------------------------------------------------

HS_DEMO_QUERIES = [
    "BM25 Okapi",
    "how do transformers handle long sequences",
    "Mixtral 8x7B sparse mixture of experts",
    "retrieval augmented generation knowledge",
    "Karpukhin dense passage",
]

with gr.Blocks(title="Paper POCs — AI Engineering", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Paper POCs — AI Engineering\n"
        "Practical reproductions of high-relevance AI/ML papers. "
        "Each tab is a runnable demo of a technique directly applicable to RAG, retrieval, or LLM evaluation.\n\n"
        "| Paper | Score | Tab |\n"
        "|-------|-------|-----|\n"
        "| Query-Adaptive Hybrid Search (DAT/vstash) | 🔴 10/10 | Hybrid Search |\n"
        "| CAD Score: Context-Aware Semantic Diversity | 🔴 9/10 | CAD Score |\n"
        "| Dynamic Sub-Sequence Warping (DSSW) | 🔴 9/10 | DSSW |\n"
        "| Synergistic Engine: CEP + Declarative Rules | 🔴 8/10 | Synergistic Engine |\n"
        "| Attribution Crisis: LLM Search Attribution | 🔴 8/10 | Attribution Crisis |\n"
        "| OLR-WAA: Adaptive Drift-Resilient Online Regression | 🔴 8/10 | OLR-WAA |"
    )

    with gr.Tabs():
        # ── Tab 1: Hybrid Search ──────────────────────────────────────────
        with gr.Tab("🔍 Hybrid Search"):
            gr.Markdown(
                "## Query-Adaptive Hybrid Search\n"
                "**Paper: DAT (arXiv:2503.23013) + vstash (arXiv:2604.15484) — Score 10/10**\n\n"
                "Each query gets its own optimal `alpha` weight between BM25 (sparse) and "
                "semantic search (dense), computed from the query's mean IDF. "
                "High-IDF / rare-term queries lean sparse; common-word / conceptual queries lean dense."
            )

            with gr.Row():
                with gr.Column(scale=3):
                    hs_query = gr.Textbox(
                        label="Query",
                        placeholder="Try a keyword query or a semantic question…",
                        lines=1,
                    )
                    gr.Examples(HS_DEMO_QUERIES, inputs=hs_query, label="Example queries")

                with gr.Column(scale=1):
                    hs_fusion = gr.Dropdown(
                        ["Score fusion", "RRF fusion"], value="Score fusion", label="Fusion method"
                    )
                    hs_topk   = gr.Slider(1, 10, value=5, step=1, label="Top-K results")
                    hs_auto   = gr.Checkbox(value=True, label="Auto alpha (IDF-adaptive)")
                    hs_alpha  = gr.Slider(
                        0.0, 1.0, value=0.5, step=0.05,
                        label="Manual alpha (0=dense, 1=sparse)", interactive=True
                    )

            hs_btn = gr.Button("Search", variant="primary")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Results")
                    hs_results = gr.Markdown()
                with gr.Column():
                    gr.Markdown("### Alpha Analysis")
                    hs_alpha_md = gr.Markdown()
                    gr.Markdown("### Fixed vs Adaptive Comparison")
                    hs_compare  = gr.Markdown()

            hs_btn.click(
                fn=hs_search,
                inputs=[hs_query, hs_fusion, hs_topk, hs_alpha, hs_auto],
                outputs=[hs_results, hs_alpha_md, hs_compare],
            )
            hs_query.submit(
                fn=hs_search,
                inputs=[hs_query, hs_fusion, hs_topk, hs_alpha, hs_auto],
                outputs=[hs_results, hs_alpha_md, hs_compare],
            )

        # ── Tab 2: CAD Score ──────────────────────────────────────────────
        with gr.Tab("📊 CAD Score"):
            gr.Markdown(
                "## CAD Score: Context-Aware Semantic Diversity\n"
                "**Paper: Kim et al. 2025 — PeerJ Computer Science — Score 9/10**\n\n"
                "Measures how *meaningfully* diverse a set of LLM outputs are, relative to the prompt. "
                "Unlike raw embedding spread, CAD penalises responses that merely paraphrase the question."
            )

            with gr.Tabs():
                with gr.Tab("Live (Anthropic API)"):
                    gr.Markdown(
                        "Generate N responses to the same prompt and immediately score their diversity. "
                        + ("⚠️ **ANTHROPIC_API_KEY not set** — set it to enable this tab."
                           if not HAS_API_KEY else "✅ API key detected.")
                    )
                    with gr.Row():
                        cad_live_prompt = gr.Textbox(
                            label="Prompt", lines=2,
                            placeholder="Give me 5 startup ideas in the AI space."
                        )
                        with gr.Column():
                            cad_n_resp   = gr.Slider(2, 8, value=5, step=1, label="Number of responses")
                            cad_creative = gr.Checkbox(value=False, label="Use creativity system prompt")
                    gr.Examples(EXAMPLE_PROMPTS, inputs=cad_live_prompt)
                    cad_live_btn = gr.Button("Generate & Score", variant="primary")

                    with gr.Row():
                        cad_live_scores  = gr.Markdown(label="Scores")
                        cad_live_explain = gr.Markdown(label="CAD Explanation")
                    with gr.Row():
                        cad_live_pairs = gr.Markdown(label="Pairwise Distances")
                        cad_live_resps = gr.Markdown(label="Generated Responses")

                    cad_live_btn.click(
                        fn=cad_evaluate_live,
                        inputs=[cad_live_prompt, cad_n_resp, cad_creative],
                        outputs=[cad_live_scores, cad_live_pairs, cad_live_explain, cad_live_resps],
                    )

                with gr.Tab("Manual (paste your own)"):
                    gr.Markdown("Paste your own responses (one per line) — no API key needed.")
                    with gr.Row():
                        cad_manual_prompt = gr.Textbox(
                            label="Prompt (context for CAD)", lines=2,
                            placeholder="The prompt that was given to the LLM"
                        )
                        cad_manual_resps = gr.Textbox(
                            label="Responses (one per line)", lines=8, value=EXAMPLE_MANUAL
                        )
                    cad_manual_btn = gr.Button("Compute CAD Score", variant="primary")

                    with gr.Row():
                        cad_man_scores  = gr.Markdown(label="Scores")
                        cad_man_explain = gr.Markdown(label="CAD Explanation")
                    with gr.Row():
                        cad_man_pairs = gr.Markdown(label="Pairwise Distances")
                        cad_man_resps = gr.Markdown(label="Responses")

                    cad_manual_btn.click(
                        fn=cad_evaluate_manual,
                        inputs=[cad_manual_prompt, cad_manual_resps],
                        outputs=[cad_man_scores, cad_man_pairs, cad_man_explain, cad_man_resps],
                    )

            gr.Markdown(
                "---\n"
                "**Interpreting scores:**  \n"
                "CAD > 0.15 → good diversity · CAD 0.05–0.15 → moderate · CAD < 0.05 → low"
            )

        # ── Tab 3: DSSW ──────────────────────────────────────────────────
        with gr.Tab("📈 DSSW"):
            gr.Markdown(
                "## Dynamic Sub-Sequence Warping (DSSW)\n"
                "**Paper: Score 9/10 — Representation-Based Similarity for Long Time Series**\n\n"
                "DSSW segments each series into overlapping sub-sequences and runs a DTW-style "
                "dynamic-programming alignment *at the sub-sequence level*. This makes it "
                "tractable for long series while staying sensitive to local shape patterns — "
                "useful for comparing embedding-drift trajectories, agent-state histories, or "
                "sequential retrieval-score runs."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    dssw_ex_selector = gr.Dropdown(
                        list(DSSW_EXAMPLES.keys()),
                        value=list(DSSW_EXAMPLES.keys())[0],
                        label="Load example",
                    )
                    dssw_load_btn = gr.Button("Load example", size="sm")
                with gr.Column(scale=1):
                    dssw_subseq_len = gr.Slider(3, 20, value=8, step=1,
                                                label="Sub-sequence length")
                    dssw_stride     = gr.Slider(1, 10, value=4, step=1,
                                                label="Stride")

            with gr.Row():
                dssw_ts1 = gr.Textbox(
                    label="Series 1 (comma-separated numbers)",
                    lines=2,
                    value=str(list(DSSW_EXAMPLES.values())[0][0])[1:-1],
                )
                dssw_ts2 = gr.Textbox(
                    label="Series 2 (comma-separated numbers)",
                    lines=2,
                    value=str(list(DSSW_EXAMPLES.values())[0][1])[1:-1],
                )

            dssw_btn = gr.Button("Compute DSSW", variant="primary")

            dssw_summary = gr.Markdown(label="Distance metrics")
            dssw_plot    = gr.Plot(label="Visualisation")

            dssw_load_btn.click(
                fn=dssw_load_example,
                inputs=[dssw_ex_selector],
                outputs=[dssw_ts1, dssw_ts2],
            )
            dssw_btn.click(
                fn=dssw_run,
                inputs=[dssw_ts1, dssw_ts2, dssw_subseq_len, dssw_stride],
                outputs=[dssw_summary, dssw_plot],
            )

            gr.Markdown(
                "---\n"
                "**DSSW vs DTW:**  \n"
                "Standard DTW aligns individual points — O(n·m) per pair. "
                "DSSW reduces this to O(n/stride · m/stride) at the sub-sequence level, "
                "trading fine-grained alignment for speed on long series. "
                "Lower DSSW distance = more structurally similar trajectories."
            )

        # ── Tab 4: Synergistic Engine ─────────────────────────────────────
        with gr.Tab("⚡ Synergistic Engine"):
            gr.Markdown(
                "## Synergistic Engine: CEP + Declarative Rules for RAG\n"
                "**Paper: 'A synergistic engine paradigm for real-time, context-aware decision-making: "
                "integrating declarative processes and event streams' — Score 8/10**\n\n"
                "Combines **Complex Event Processing (CEP)** with a **declarative rule engine** to make "
                "context-aware retrieval and generation decisions in real time. As an agent emits events "
                "(queries, tool calls, context retrievals, errors), the engine pattern-matches against "
                "a set of declarative IF-THEN rules to select the right retrieval or generation strategy — "
                "no hard-coded if/else chains needed.\n\n"
                "**RAG applicability:** Query routing (sparse vs dense vs hybrid), reranking triggers, "
                "error escalation, cache-hit detection, and response-quality monitoring."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    se_preset = gr.Dropdown(
                        list(SE_EVENT_PRESETS.keys()),
                        value=list(SE_EVENT_PRESETS.keys())[0],
                        label="Emit an event",
                        info="Select a simulated event to inject into the stream",
                    )
                    with gr.Row():
                        se_emit_btn  = gr.Button("▶ Emit Event", variant="primary")
                        se_reset_btn = gr.Button("🔄 Reset Stream", variant="secondary")

                with gr.Column(scale=1):
                    gr.Markdown(
                        "### How it works\n"
                        "1. Select an event preset and click **Emit Event**\n"
                        "2. The event is appended to the sliding window stream\n"
                        "3. All declarative rules are evaluated against the full stream\n"
                        "4. The highest-priority rule that fires determines the decision\n"
                        "5. Build up a multi-event sequence to trigger complex patterns"
                    )

            se_event_store = gr.State([])

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Current Decision")
                    se_decision_md = gr.Markdown("_Emit an event to see a decision._")
                with gr.Column():
                    gr.Markdown("### Rule Evaluation")
                    se_rule_table  = gr.Markdown("_Rules evaluated here after first event._")

            gr.Markdown("### Event Stream (sliding window)")
            se_stream_md = gr.Markdown("_empty_")

            gr.Markdown("### Session Log")
            se_log_md = gr.Markdown("_No events emitted yet._")

            se_emit_btn.click(
                fn=se_emit_event,
                inputs=[se_preset, se_event_store],
                outputs=[se_decision_md, se_rule_table, se_stream_md, se_event_store],
            ).then(
                fn=se_format_log,
                inputs=[se_event_store],
                outputs=[se_log_md],
            )
            se_reset_btn.click(
                fn=se_reset,
                inputs=[se_event_store],
                outputs=[se_decision_md, se_rule_table, se_stream_md, se_event_store],
            ).then(
                fn=se_format_log,
                inputs=[se_event_store],
                outputs=[se_log_md],
            )

            gr.Markdown(
                "---\n"
                "**Rule priorities (higher = evaluated first):**  \n"
                "P10: Error escalation · P9: Cache hit detection · P8: Sparse/dense routing  \n"
                "P7: Low-confidence regeneration · P6: Post-tool reranking · P5: Hybrid fallback\n\n"
                "**Key insight from the paper:** Separating *what to decide* (declarative rules) from "
                "*how decisions execute* (action handlers) makes the system inspectable, auditable, "
                "and easy to extend — unlike imperative if/else routing logic embedded in agent code."
            )

        # ── Tab 5: Attribution Crisis ─────────────────────────────────────
        with gr.Tab("📜 Attribution Crisis"):  # noqa: E501 (kept for readability)
            gr.Markdown(
                "## Attribution Crisis: LLM Search Attribution Scorer\n"
                "**Paper: 'The attribution crisis in LLM search results: Estimating ecosystem exploitation' — Score 8/10**\n\n"
                "When LLMs generate answers from retrieved documents, they often reproduce substantial portions "
                "of source content *without citing it* — exploiting the ecosystem. This tool quantifies that gap "
                "using **lexical overlap** (n-gram coverage), **semantic similarity** (TF-IDF cosine), and "
                "**citation recall** against the retrieved document set.\n\n"
                "**RAG applicability:** Audit your RAG pipeline's attribution behaviour. Identify responses "
                "that reproduce source content without credit, compute exploitation risk, and enforce citation policies."
            )

            with gr.Tabs():
                with gr.Tab("Preset Examples"):
                    gr.Markdown("Select a preset RAG response to see how attribution is scored.")
                    ac_preset_dd = gr.Dropdown(
                        list(AC_PRESET_RESPONSES.keys()),
                        value=list(AC_PRESET_RESPONSES.keys())[0],
                        label="RAG response preset",
                    )
                    ac_preset_btn = gr.Button("Score Attribution", variant="primary")

                    with gr.Row():
                        ac_pre_summary = gr.Markdown(label="Attribution Summary")
                        ac_pre_explain = gr.Markdown(label="Score Explanation")
                    ac_pre_sources = gr.Markdown(label="Per-Source Breakdown")

                    ac_preset_btn.click(
                        fn=ac_run_preset,
                        inputs=[ac_preset_dd],
                        outputs=[ac_pre_summary, ac_pre_sources, ac_pre_explain],
                    )

                with gr.Tab("Custom Input"):
                    gr.Markdown(
                        "Paste your own query, RAG response, and cited document IDs "
                        f"to score against the built-in corpus. Available doc IDs: "
                        f"`{', '.join(d.id for d in AC_CORPUS)}`"
                    )
                    with gr.Row():
                        ac_cust_query  = gr.Textbox(label="Query", lines=1,
                                                    placeholder="What question was asked?")
                        ac_cust_ids    = gr.Textbox(label="Cited doc IDs (space or comma separated)",
                                                    lines=1, placeholder="ac1 ac2")
                    ac_cust_answer = gr.Textbox(label="RAG-generated answer", lines=6,
                                                placeholder="Paste the LLM response here…")
                    ac_cust_btn = gr.Button("Score Attribution", variant="primary")

                    with gr.Row():
                        ac_cust_summary = gr.Markdown(label="Attribution Summary")
                        ac_cust_explain = gr.Markdown(label="Score Explanation")
                    ac_cust_sources = gr.Markdown(label="Per-Source Breakdown")

                    ac_cust_btn.click(
                        fn=ac_run_custom,
                        inputs=[ac_cust_query, ac_cust_answer, ac_cust_ids],
                        outputs=[ac_cust_summary, ac_cust_sources, ac_cust_explain],
                    )

            gr.Markdown(
                "---\n"
                "**Exploitation risk levels:**  \n"
                "🔴 HIGH (> 0.25): Substantial source content reproduced without citation  \n"
                "🟡 MEDIUM (0.10–0.25): Moderate reuse, partial attribution  \n"
                "🟢 LOW (< 0.10): Minimal reuse or fully attributed\n\n"
                "**Key insight from the paper:** LLM-powered search systems systematically "
                "reproduce publisher content without attribution, creating an exploitation asymmetry — "
                "users receive synthesised answers while original authors lose traffic and credit. "
                "This scorer makes that risk measurable and actionable at the pipeline level."
            )

        # ── Tab 6: OLR-WAA ───────────────────────────────────────────────
        with gr.Tab("📉 OLR-WAA"):
            gr.Markdown(
                "## OLR-WAA: Adaptive & Drift-Resilient Online Regression\n"
                "**Paper: 'OLR-WAA: Adaptive and Drift-Resilient Online Regression with Dynamic Weighted Averaging' — Score 8/10**\n\n"
                "Maintains a pool of online linear regressors weighted by their recent prediction accuracy. "
                "When the data distribution drifts (e.g. a sudden slope change), WAA down-weights stale experts "
                "and promotes accurate ones — all without retraining from scratch.\n\n"
                "**RAG applicability:** Adaptive retrieval score thresholds, embedding drift tracking, "
                "dynamic alpha tuning for hybrid search as corpus statistics change over time."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Data stream")
                    olr_n_steps     = gr.Slider(50, 400, value=200, step=10,  label="Stream length (steps)")
                    olr_drift_type  = gr.Dropdown(
                        ["abrupt", "gradual", "linear"],
                        value="abrupt", label="Drift type",
                        info="'linear' = no drift; 'abrupt' = sudden slope change; 'gradual' = smooth transition"
                    )
                    olr_drift_point = gr.Slider(20, 380, value=100, step=5,   label="Drift point (step)")
                    olr_noise_std   = gr.Slider(0.1, 3.0, value=0.5, step=0.1, label="Noise σ")
                    olr_a_before    = gr.Slider(-5.0, 5.0, value=1.0, step=0.5, label="Slope before drift")
                    olr_a_after     = gr.Slider(-5.0, 5.0, value=-2.0, step=0.5, label="Slope after drift")

                with gr.Column(scale=1):
                    gr.Markdown("### OLR-WAA hyper-parameters")
                    olr_n_experts   = gr.Slider(2, 10, value=5, step=1,       label="Number of experts")
                    olr_eta         = gr.Slider(0.01, 2.0, value=0.5, step=0.05,
                                                label="WAA η (weight update rate)",
                                                info="Higher = faster expert switching")
                    olr_ff          = gr.Slider(0.80, 1.00, value=0.98, step=0.01,
                                                label="Forgetting factor λ",
                                                info="Lower = older losses discounted faster → faster adaptation")
                    olr_btn = gr.Button("Run Simulation", variant="primary")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Metrics")
                    olr_summary = gr.Markdown()
                with gr.Column():
                    gr.Markdown("### Algorithm Explanation")
                    olr_explain = gr.Markdown()

            olr_plot = gr.Plot(label="Simulation Plots")

            olr_btn.click(
                fn=olr_run,
                inputs=[
                    olr_n_steps, olr_drift_type, olr_drift_point,
                    olr_noise_std, olr_a_before, olr_a_after,
                    olr_n_experts, olr_eta, olr_ff,
                ],
                outputs=[olr_summary, olr_explain, olr_plot],
            )

            gr.Markdown(
                "---\n"
                "**Key insight from the paper:** Static models fail silently when the data distribution "
                "shifts — they keep predicting using stale weights and the error silently grows. "
                "OLR-WAA detects this through rising prediction losses and dynamically re-weights "
                "experts, achieving significantly lower post-drift MAE with no manual retraining trigger.\n\n"
                "**Forgetting factor guide:** λ=0.99 → slow adaptation (stable data); "
                "λ=0.90 → fast adaptation (volatile data). "
                "η controls the sharpness of expert selection — higher η → winner-takes-most behaviour."
            )

if __name__ == "__main__":
    demo.launch()
