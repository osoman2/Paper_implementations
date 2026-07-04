"""
Paper POCs — multi-tab Gradio demo.

Tabs:
  1. Query-Adaptive Hybrid Search (DAT / vstash) — score 10/10
  2. CAD Score: Context-Aware Semantic Diversity — score 9/10
  3. Dynamic Sub-Sequence Warping (DSSW) — score 9/10
  4. Synergistic Engine (CEP + Declarative Rules for RAG) — score 8/10
  5. Attribution Crisis (LLM Search Attribution Scorer) — score 8/10
  6. OLR-WAA: Adaptive Drift-Resilient Online Regression — score 8/10
  7. Bi-Stage Adaptive RAG (Group Recommendation → RAG reranking) — score 8/10
  8. Opinion Maximization with Fairness (Social Network Influence) — score 8/10
  9. TopoMAS: LLM-Driven Topological Materials Multi-Agent System — score 8/10
 10. Label Scarcity: Semi-Supervised Pseudo-Labeling for RAG Query Routing — score 8/10
 11. Extreme Event AI: Reliability Calibration for RAG on Edge-Case Queries — score 8/10
 12. GrepSeek DCI: Direct Corpus Interaction — Hybrid Search for AI Agents — score 9/10
 13. DeepSeek-Lattice-KG: Knowledge Graph Augmented RAG — score 8/10
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

# ---------------------------------------------------------------------------
# Tab 12 — GrepSeek DCI: Direct Corpus Interaction
# ---------------------------------------------------------------------------
from grepseek_dci.engine import (
    semantic_search as dci_semantic_search,
    dci_search,
    hybrid_search as dci_hybrid_search,
    format_results_md,
)
from grepseek_dci.corpus import CORPUS as DCI_CORPUS

# ---------------------------------------------------------------------------
# Tab 13 — DeepSeek-Lattice-KG: Knowledge Graph Augmented RAG
# ---------------------------------------------------------------------------
from deepseek_lattice_kg.query_engine import (
    kg_rag_search,
    bm25_search,
    build_graph_data,
)
from deepseek_lattice_kg.knowledge_graph import Edge as KGEdge

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

# ---------------------------------------------------------------------------
# Tab 7 — Bi-Stage Adaptive RAG
# ---------------------------------------------------------------------------
from bi_stage_rag import MultiAspectQuery, Stage1Retriever, AdaptiveReranker
# ---------------------------------------------------------------------------
# Tab 8 — Opinion Maximization with Fairness
# ---------------------------------------------------------------------------
from opinion_maximization import SocialGraph, simulate_cascade, greedy_maximize

# ---------------------------------------------------------------------------
# Tab 9 — TopoMAS: LLM-Driven Topological Materials Multi-Agent System
# ---------------------------------------------------------------------------
from topomas import MultiAgentOrchestrator

# ---------------------------------------------------------------------------
# Tab 10 — Label Scarcity: Semi-Supervised Pseudo-Labeling
# ---------------------------------------------------------------------------
from label_scarcity import SemiSupervisedLabeler, LabeledExample, PseudoLabelResult

# ---------------------------------------------------------------------------
# Tab 11 — Extreme Event AI: Reliability Calibration
# ---------------------------------------------------------------------------
from extreme_event_ai import ModelComparison, BUILTIN_MODELS as EEA_MODELS
from extreme_event_ai import CalibrationReport  # noqa: F401 (used in type hints)

_topo_orchestrator = MultiAgentOrchestrator()

_OM_GRAPH = SocialGraph.sample_network(n_majority=15, n_minority=5, seed=42)
_OM_POS   = _OM_GRAPH.layout_positions()
from bi_stage_rag.retriever import Document as BSDocument

BS_CORPUS = [
    BSDocument("d1",  "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"),
    BSDocument("d2",  "Attention Is All You Need — Transformer architecture by Vaswani et al."),
    BSDocument("d3",  "GPT-4 Technical Report — OpenAI large multimodal language model"),
    BSDocument("d4",  "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al.)"),
    BSDocument("d5",  "Dense Passage Retrieval for Open-Domain Question Answering (Karpukhin et al.)"),
    BSDocument("d6",  "BM25 Okapi probabilistic ranking model for information retrieval"),
    BSDocument("d7",  "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction"),
    BSDocument("d8",  "LlamaIndex is a data framework for building LLM-powered retrieval applications"),
    BSDocument("d9",  "LangChain provides tools to compose language model chains and agents"),
    BSDocument("d10", "Vector databases like Qdrant, Pinecone, and Weaviate store embeddings at scale"),
    BSDocument("d11", "Elasticsearch supports full-text search using inverted index and BM25 scoring"),
    BSDocument("d12", "Fine-tuning sentence transformers improves domain-specific semantic similarity"),
    BSDocument("d13", "Reciprocal Rank Fusion combines ranked lists from heterogeneous retrieval systems"),
    BSDocument("d14", "Hybrid search blends sparse BM25 and dense vector retrieval for better recall"),
    BSDocument("d15", "Cross-encoder reranking rescores candidate documents with a BERT-based model"),
    BSDocument("d16", "Mixtral 8x7B — sparse mixture of experts model for efficient language generation"),
    BSDocument("d17", "Anthropic Claude 3 — constitutional AI approach to safe and helpful assistants"),
    BSDocument("d18", "Tree of Thoughts prompting enables LLMs to explore multiple reasoning paths"),
    BSDocument("d19", "Self-RAG: Learning to retrieve, generate, and critique with reflection tokens"),
    BSDocument("d20", "Multi-vector retrieval with ColBERT outperforms single-vector bi-encoder models"),
]

_bs_retriever = Stage1Retriever(BS_CORPUS)
_bs_reranker  = AdaptiveReranker()
_bs_doc_map   = {d.id: d for d in BS_CORPUS}

BS_PRESETS = {
    "Retrieval + efficiency": (
        "retrieval augmented generation pipeline",
        ["fast efficient low latency", "dense vector search embedding"],
    ),
    "Transformer architecture depth": (
        "transformer attention mechanism",
        ["BERT fine-tuning language model", "multi-head self-attention layers"],
    ),
    "Hybrid search strategies": (
        "combining sparse and dense retrieval",
        ["BM25 keyword scoring", "semantic similarity cosine distance"],
    ),
    "LLM agents and tools": (
        "LLM agent tool use reasoning",
        ["chain of thought prompting", "self-critique reflection tokens"],
    ),
}


def bs_run(
    query: str,
    aspect1: str,
    aspect2: str,
    top_k: int,
    temperature: float,
    use_fixed_weights: bool,
    fixed_w1: float,
    fixed_w2: float,
):
    aspects = [a.strip() for a in [aspect1, aspect2] if a.strip()]
    if not query.strip():
        return "Enter a query first.", "", "", ""
    if not aspects:
        aspects = ["relevance", "quality"]

    mq = MultiAspectQuery(text=query, aspects=aspects)
    candidates = _bs_retriever.generate_candidates(mq, top_k_per_aspect=max(top_k, 8))

    fixed_w = None
    if use_fixed_weights and aspects:
        total = fixed_w1 + fixed_w2 + 1e-9
        fixed_w = [fixed_w1 / total, fixed_w2 / total]
        if len(aspects) == 1:
            fixed_w = [1.0]

    results = _bs_reranker.rerank(
        candidates, query, aspects, _bs_doc_map,
        temperature=temperature, fixed_weights=fixed_w,
    )[:top_k]

    # Stage 1 table: candidates per aspect
    s1_rows = []
    aspect_names = ["main_query"] + [f"aspect_{i+1}" for i in range(len(aspects))]
    for cand in sorted(candidates, key=lambda c: c.max_score, reverse=True)[:top_k]:
        scores_str = " | ".join(
            f"{an}={cand.aspect_scores.get(an, 0):.2f}" for an in aspect_names
        )
        s1_rows.append(f"| {cand.doc_id} | {scores_str} | {_bs_doc_map[cand.doc_id].text[:55]}… |")

    aspect_header = " | ".join(aspect_names)
    s1_table = (
        f"| Doc ID | {aspect_header} | Text |\n"
        + "|--------|" + "--------|" * len(aspect_names) + "------|\n"
        + "\n".join(s1_rows)
    )

    # Stage 2 table: re-ranked results
    s2_rows = []
    for r in results:
        weight_str = " | ".join(f"{k}={v:.3f}" for k, v in r.aspect_weights.items())
        s2_rows.append(
            f"| {r.rank} | {r.doc_id} | {r.final_score:.4f} | {weight_str} | {r.text[:50]}… |"
        )

    weight_header = " | ".join(r.aspect_weights.keys()) if results else "weights"
    s2_table = (
        f"| Rank | Doc ID | Score | {weight_header} | Text |\n"
        + "|------|--------|-------|" + "-------|" * len(results[0].aspect_weights if results else {}) + "------|\n"
        + "\n".join(s2_rows)
    )

    # Attention weights explanation
    if results:
        w_display = "\n".join(
            f"- **{k}**: {v:.3f}" for k, v in results[0].aspect_weights.items()
        )
        attn_md = (
            f"### Adaptive Attention Weights (temperature={temperature})\n\n"
            f"{w_display}\n\n"
            f"**Interpretation:** Aspects more similar to the main query receive higher "
            f"attention weight. Temperature controls sharpness: low T → winner-takes-most."
        )
    else:
        attn_md = "_No results._"

    explain_md = f"""
**How Bi-Stage Adaptive RAG works (from the paper):**

1. **Stage 1 — Candidate Generation**: BM25 scores each document independently against
   the main query **and** each aspect (e.g., "fast", "dense vector"). The top-K per aspect
   are merged into a candidate pool of size ≤ K × (1 + N_aspects).

2. **Stage 2 — Adaptive Reranking**: Attention weights are computed from TF-IDF cosine
   similarity between each aspect and the main query:
   `weight_i = softmax(sim(query, aspect_i) / τ)`
   Final score = 0.5 × main_query_score + 0.5 × Σ(weight_i × aspect_i_score)

3. **Why it matters for RAG**: Multi-faceted queries (e.g., "efficient Python RAG for beginners")
   benefit from per-aspect retrieval. Adaptive weights let the dominant aspect drive ranking
   without requiring hand-tuned static weights for every query type.

**Paper insight:** Replacing fixed aggregation with a learned/dynamic attention over preference
aspects gave 5–15% NDCG improvement in group recommendation settings — analogous to per-query
alpha tuning in hybrid search.
""".strip()

    return s1_table, s2_table, attn_md, explain_md


def bs_load_preset(preset_name: str):
    q, aspects = BS_PRESETS[preset_name]
    a1 = aspects[0] if len(aspects) > 0 else ""
    a2 = aspects[1] if len(aspects) > 1 else ""
    return q, a1, a2


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

def _om_draw_graph(seeds: list[int], title: str, ax):
    pos = _OM_POS
    graph = _OM_GRAPH
    seed_set = set(seeds)

    # Draw edges
    for i in range(graph.n):
        for j in range(i + 1, graph.n):
            w = graph.adj[i, j]
            if w > 0:
                xi, yi = pos[i]
                xj, yj = pos[j]
                ax.plot([xi, xj], [yi, yj], color="#aaa", linewidth=w * 3,
                        alpha=0.4, zorder=1)

    # Draw nodes
    for nid, node in graph.nodes.items():
        x, y = pos[nid]
        color  = "#2196F3" if node.group == "majority" else "#FF9800"
        size   = 280 if nid in seed_set else 120
        marker = "*" if nid in seed_set else "o"
        ax.scatter(x, y, s=size, c=color, marker=marker, zorder=3,
                   edgecolors="black" if nid in seed_set else "none",
                   linewidths=1.5)
        ax.text(x, y + 0.18, node.label, ha="center", va="bottom",
                fontsize=7, zorder=4)

    ax.set_title(title, fontsize=10)
    ax.set_xlim(-3.0, 4.0)
    ax.set_ylim(-2.2, 2.2)
    ax.axis("off")


def om_run(k: int, fairness_budget: int, n_sims: int):
    k = max(1, min(k, 8))
    fairness_budget = max(0, min(fairness_budget, k, len(_OM_GRAPH.group_ids("minority"))))

    seeds_plain, stats_plain  = greedy_maximize(_OM_GRAPH, k=k, fairness_budget=0, n_sims=n_sims)
    seeds_fair,  stats_fair   = greedy_maximize(_OM_GRAPH, k=k, fairness_budget=fairness_budget, n_sims=n_sims)

    def badge(v: float) -> str:
        return "🟢" if v >= 0.6 else "🟡" if v >= 0.3 else "🔴"

    comparison = f"""
| Metric | Unconstrained | Fairness-constrained |
|--------|--------------|----------------------|
| Seeds selected | `{seeds_plain}` | `{seeds_fair}` |
| Mean spread | **{stats_plain['mean_spread']:.2f}** nodes | **{stats_fair['mean_spread']:.2f}** nodes |
| Majority coverage | {badge(stats_plain['majority_coverage'])} {stats_plain['majority_coverage']:.1%} | {badge(stats_fair['majority_coverage'])} {stats_fair['majority_coverage']:.1%} |
| Minority coverage | {badge(stats_plain['minority_coverage'])} {stats_plain['minority_coverage']:.1%} | {badge(stats_fair['minority_coverage'])} {stats_fair['minority_coverage']:.1%} |
| Fairness gap | {stats_plain['fairness_gap']:.1%} | **{stats_fair['fairness_gap']:.1%}** |
""".strip()

    seeds_plain_groups = [_OM_GRAPH.nodes[s].group for s in seeds_plain]
    seeds_fair_groups  = [_OM_GRAPH.nodes[s].group for s in seeds_fair]
    plain_detail = (
        f"**Seeds (unconstrained):** {seeds_plain}\n\n"
        f"Groups: {seeds_plain_groups}\n\n"
        f"Spread: {stats_plain['mean_spread']:.2f} / {_OM_GRAPH.n} nodes  \n"
        f"Coverage — Majority: {stats_plain['majority_coverage']:.1%}  · "
        f"Minority: {stats_plain['minority_coverage']:.1%}"
    )
    fair_detail = (
        f"**Seeds (fairness budget={fairness_budget}):** {seeds_fair}\n\n"
        f"Groups: {seeds_fair_groups}\n\n"
        f"Spread: {stats_fair['mean_spread']:.2f} / {_OM_GRAPH.n} nodes  \n"
        f"Coverage — Majority: {stats_fair['majority_coverage']:.1%}  · "
        f"Minority: {stats_fair['minority_coverage']:.1%}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#f9f9f9")
    _om_draw_graph(seeds_plain, f"Unconstrained (k={k})", axes[0])
    _om_draw_graph(seeds_fair,  f"Fairness-constrained (budget={fairness_budget})", axes[1])

    # Legend
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(color="#2196F3", label="Majority group"),
        mpatches.Patch(color="#FF9800", label="Minority group"),
        plt.scatter([], [], marker="*", s=180, c="gray", edgecolors="black",
                    linewidths=1.5, label="Seed node"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    return comparison, plain_detail, fair_detail, fig


# ---------------------------------------------------------------------------
# Tab 10 helpers — Label Scarcity / Semi-Supervised Pseudo-Labeling
# ---------------------------------------------------------------------------

LS_PRESETS = {
    "RAG Query Routing (factual vs conceptual)": {
        "labeled": [
            LabeledExample("What is the capital of France?", "factual"),
            LabeledExample("Who invented the telephone?", "factual"),
            LabeledExample("When was the Eiffel Tower built?", "factual"),
            LabeledExample("How does photosynthesis work conceptually?", "conceptual"),
            LabeledExample("Explain the philosophy behind transformer attention.", "conceptual"),
            LabeledExample("What is the intuition behind embeddings?", "conceptual"),
        ],
        "unlabeled": [
            "Who wrote Pride and Prejudice?",
            "How do neural networks learn representations?",
            "What year did World War II end?",
            "Why do language models hallucinate?",
            "How many planets are in the solar system?",
            "What is the difference between RAG and fine-tuning?",
            "What is the boiling point of water?",
            "Explain the role of attention in LLMs.",
            "When was Python first released?",
            "How does contrastive learning work?",
            "Who is the CEO of OpenAI?",
            "What is the intuition behind BM25 scoring?",
            "How many bones are in the human body?",
            "Why does sparse retrieval work well for keyword queries?",
            "What is the height of Mount Everest?",
        ],
    },
    "Intent Classification (retrieve vs generate vs summarize)": {
        "labeled": [
            LabeledExample("Find all documents about transformer architecture.", "retrieve"),
            LabeledExample("Search for papers on hybrid search.", "retrieve"),
            LabeledExample("Look up the definition of cosine similarity.", "retrieve"),
            LabeledExample("Write a summary of the RAG pipeline.", "generate"),
            LabeledExample("Create an explanation of BM25 for a beginner.", "generate"),
            LabeledExample("Compose a blog post about vector databases.", "generate"),
            LabeledExample("Condense this 10-page paper into 3 bullet points.", "summarize"),
            LabeledExample("Give me the key takeaways from this document.", "summarize"),
            LabeledExample("Summarize the findings from the literature review.", "summarize"),
        ],
        "unlabeled": [
            "Get me all papers on LLM evaluation metrics.",
            "Draft a technical overview of dense retrieval.",
            "Distill the main points from this report.",
            "Locate any documents mentioning ColBERT.",
            "Write a comparison between BM25 and DPR.",
            "Boil down this paper to its essential contribution.",
            "Fetch the most recent papers on speculative decoding.",
            "Generate a step-by-step tutorial on RAG.",
            "Compress the abstract into one sentence.",
            "Find documents where the author discusses fairness.",
            "Build a narrative around the retrieved context.",
            "Extract the key claims from the methodology section.",
        ],
    },
    "Retrieval Strategy (sparse vs dense vs hybrid)": {
        "labeled": [
            LabeledExample("BM25 term frequency inverse document frequency keyword exact match", "sparse"),
            LabeledExample("Okapi BM25 probabilistic ranking token overlap", "sparse"),
            LabeledExample("inverted index lexical search vocabulary lookup", "sparse"),
            LabeledExample("semantic similarity cosine distance embedding vector", "dense"),
            LabeledExample("sentence transformer BERT encoder dense passage retrieval", "dense"),
            LabeledExample("neural embedding nearest neighbour faiss similarity", "dense"),
            LabeledExample("blend sparse dense retrieval fusion alpha weight", "hybrid"),
            LabeledExample("reciprocal rank fusion combine BM25 semantic score", "hybrid"),
            LabeledExample("adaptive alpha mixing retrieval strategies ensemble", "hybrid"),
        ],
        "unlabeled": [
            "keyword exact match term lookup TF-IDF scoring",
            "neural encoder deep embedding representation similarity",
            "combined retrieval system alpha parameter tuning",
            "BERT sentence encoding passage similarity search",
            "token presence vocabulary boolean search",
            "weighted ensemble retrieval methods fusion score",
            "approximate nearest neighbour search HNSW index",
            "n-gram overlap matching bag of words",
            "query-adaptive blending of retrieval strategies",
            "document frequency corpus statistics IDF weight",
            "cross-encoder reranking neural scoring pipeline",
            "linear combination sparse dense results merging",
        ],
    },
}


def ls_run(
    preset_name: str,
    confidence_threshold: float,
    max_rounds: int,
    k_neighbors: int,
    extra_unlabeled: str,
) -> tuple[str, str, str, str]:
    preset = LS_PRESETS[preset_name]
    labeled = preset["labeled"]
    unlabeled = list(preset["unlabeled"])

    for line in extra_unlabeled.strip().splitlines():
        line = line.strip()
        if line:
            unlabeled.append(line)

    labeler = SemiSupervisedLabeler(
        confidence_threshold=confidence_threshold,
        max_rounds=int(max_rounds),
        k=int(k_neighbors),
    )
    results, round_stats = labeler.fit_predict(labeled, unlabeled)

    # Seed table
    seed_rows = [
        f"| {e.label} | {e.text} |"
        for e in labeled
    ]
    seed_table = (
        "| Label | Text |\n|-------|------|\n" + "\n".join(seed_rows)
    )

    # Results table
    res_rows = []
    for r in results:
        icon = "✅" if r.accepted else "❌"
        res_rows.append(
            f"| {icon} | {r.text[:55]} | **{r.predicted_label}** | "
            f"{r.confidence:.3f} | {r.nearest_neighbor_text[:40]}… |"
        )
    res_table = (
        "| Status | Unlabeled Text | Pseudo-Label | Confidence | Nearest Seed |\n"
        "|--------|---------------|--------------|------------|---------------|\n"
        + "\n".join(res_rows)
    )

    # Round stats
    stat_rows = []
    for s in round_stats:
        dist_str = ", ".join(f"{k}={v}" for k, v in s["label_distribution"].items())
        stat_rows.append(
            f"| {s['round']} | {s['accepted']} | {s['remaining']} | "
            f"{s['pool_size']} | {dist_str} |"
        )
    if stat_rows:
        stats_md = (
            "| Round | Accepted | Remaining | Pool size | Label distribution |\n"
            "|-------|----------|-----------|-----------|--------------------|\n"
            + "\n".join(stat_rows)
        )
    else:
        stats_md = "_No rounds completed._"

    n_accepted = sum(1 for r in results if r.accepted)
    n_total = len(results)
    n_seeds = len(labeled)
    expansion = (n_accepted / n_seeds * 100) if n_seeds else 0

    explain_md = f"""
**How Semi-Supervised Pseudo-Labeling works (from the paper):**

1. **Seed labeled pool** — Start with {n_seeds} labeled examples (scarce annotations).
2. **TF-IDF embedding** — All texts (labeled + unlabeled) encoded into TF-IDF vectors.
3. **k-NN assignment** — Each unlabeled item gets a pseudo-label via weighted majority
   vote over its {k_neighbors} nearest labeled neighbors.
4. **Confidence filtering** — Only predictions with confidence ≥ {confidence_threshold} are accepted.
   Accepted items join the labeled pool and improve future rounds.
5. **Iterative expansion** — Process repeats for up to {int(max_rounds)} rounds, growing
   the pool from {n_seeds} seeds → {n_seeds + n_accepted} labeled items (+{expansion:.0f}%).

**Results:** {n_accepted}/{n_total} unlabeled items pseudo-labeled with confidence ≥ {confidence_threshold}.

**Why this matters for AI engineering:**
- **Query routing in RAG**: Classify incoming queries (factual/semantic/hybrid) with
  just a handful of labeled examples per class — no large annotation budget needed.
- **Intent detection**: Label queries for tool-calling agents with 5–10 seeds per intent.
- **Embedding fine-tuning data**: Generate pseudo-labeled pairs to expand training sets
  for sentence transformer fine-tuning when gold labels are scarce.
""".strip()

    return seed_table, res_table, stats_md, explain_md


HS_DEMO_QUERIES = [
    "BM25 Okapi",
    "how do transformers handle long sequences",
    "Mixtral 8x7B sparse mixture of experts",
    "retrieval augmented generation knowledge",
    "Karpukhin dense passage",
]


# ---------------------------------------------------------------------------
# Tab 9 helpers — TopoMAS
# ---------------------------------------------------------------------------
def topomas_run(query: str, top_k: int) -> tuple[str, str, str]:
    if not query.strip():
        return "", "", ""
    trace = _topo_orchestrator.run(query.strip(), top_k=int(top_k))
    agent_trace_md = trace.to_markdown_trace()
    answer_md = trace.final_answer

    # Build property table
    prop_rows = []
    for v in trace.validation_verdicts:
        formula = v.get("formula", "?")
        topo = "Yes" if v.get("is_topological") else "No"
        cls = v.get("topo_class", "—")
        gap = v.get("band_gap_eV", "—")
        inv = v.get("topological_invariant", "—")
        inv_type = v.get("invariant_type", "—")
        stab = v.get("stability", "—")
        verdict = v.get("verdict", "—")
        prop_rows.append(
            f"| {formula} | {topo} | {cls} | {gap} | {inv_type}={inv} | {stab} | {verdict} |"
        )

    if prop_rows:
        prop_table = (
            "| Formula | Topological | Class | Gap (eV) | Invariant | Stability | Verdict |\n"
            "|---------|-------------|-------|----------|-----------|-----------|----------|\n"
            + "\n".join(prop_rows)
        )
    else:
        prop_table = "_No property data available for this query._"

    return agent_trace_md, prop_table, answer_md


TOPO_DEMO_QUERIES = [
    "topological insulators for quantum computing",
    "Bi2Se3 properties",
    "Weyl semimetal applications",
    "find stable topological materials",
    "suggest new topological material",
    "validate topological properties of MnBi2Te4",
    "compare HgTe and Bi2Te3",
]


# ---------------------------------------------------------------------------
# Tab 11 helpers — Extreme Event AI: Reliability Calibration
# ---------------------------------------------------------------------------

EEA_CORPUS = [
    # Normal queries — common vocabulary, average length
    "what is retrieval augmented generation",
    "how does BM25 work",
    "explain embeddings in vector databases",
    "what is a language model",
    "how to fine-tune BERT",
    "explain cosine similarity",
    "what is a transformer",
    "how does attention mechanism work",
    "difference between sparse and dense retrieval",
    "what is prompt engineering",
    "how to evaluate RAG pipelines",
    "what is semantic search",
    "how does FAISS work",
    "explain reciprocal rank fusion",
    "what is chunking in RAG",
    "how to improve retrieval recall",
    "what is a cross-encoder",
    "explain knowledge graphs in RAG",
    "how to reduce hallucinations in LLMs",
    "what are embedding models",
    # Extreme queries — rare terms, unusual phrasing, domain-specific jargon
    "Milvus HNSW ef_construction parameter tuning IVF_FLAT nprobe",
    "ColBERTv2 MaxSim late interaction token-level embedding compression",
    "FlashAttention-2 tiling algorithm memory bandwidth IO-bound computation",
    "SPLADE sparse learned representation vocabulary expansion regularisation",
    "Matryoshka Representation Learning adaptive embedding dimensionality truncation",
    "pg_vector ivfflat vs hnsw index recall tradeoff quantization",
    "Lost in the middle positional bias long context window retrieval degradation",
    "xq",  # very short out-of-vocabulary query
    "BEIR benchmark zero-shot transfer heterogeneous retrieval evaluation TREC",
    "bi-encoder dual-encoder training hard negatives in-batch negatives curriculum",
]

EEA_QUERY_PRESETS = {
    "Mixed (normal + extreme)": EEA_CORPUS,
    "Normal queries only": EEA_CORPUS[:20],
    "Extreme queries only": EEA_CORPUS[20:],
    "Custom sample": [
        "what is hybrid search",
        "BM25 Okapi probabilistic ranking term frequency saturation",
        "how does semantic search work",
        "HNSW graph index approximate nearest neighbour search Hnswlib efSearch",
        "explain RAG pipeline",
        "MMR maximal marginal relevance diversity reranking lambda threshold",
        "how to use LangChain",
        "ColBERT token-level late interaction MaxSim operator query expansion",
    ],
}

_eea_comparison = ModelComparison(corpus=EEA_CORPUS, extreme_percentile=0.75)


def eea_run(
    preset_name: str,
    selected_models: list[str],
    extreme_percentile: float,
    custom_queries_text: str,
) -> tuple[str, str, str, object]:
    """Run extreme event reliability calibration and return results."""
    if preset_name == "Custom — use text box":
        queries = [q.strip() for q in custom_queries_text.strip().splitlines() if q.strip()]
    else:
        queries = list(EEA_QUERY_PRESETS.get(preset_name, EEA_CORPUS))

    if len(queries) < 4:
        return "Need at least 4 queries.", "", "", None
    if not selected_models:
        selected_models = ["BM25 (sparse)", "Adaptive Hybrid (query-IDF α)"]

    comparison = ModelComparison(corpus=EEA_CORPUS, extreme_percentile=extreme_percentile)
    reports = comparison.run(queries, selected_models)
    classified = comparison.classify_queries(queries)

    # Query classification table
    q_rows = []
    for q, score, qtype in sorted(classified, key=lambda x: x[1], reverse=True)[:20]:
        icon = "🔴" if qtype == "extreme" else "🟢"
        q_rows.append(f"| {icon} {qtype} | {score:.3f} | {q[:70]} |")
    q_table = (
        "| Type | Extremity | Query |\n"
        "|------|-----------|-------|\n"
        + "\n".join(q_rows)
        + f"\n\n_Showing top 20 of {len(queries)} queries. "
        f"🔴 extreme: {sum(1 for _,_,t in classified if t=='extreme')} · "
        f"🟢 normal: {sum(1 for _,_,t in classified if t=='normal')}_"
    )

    # Calibration summary table
    cal_rows = []
    for mname, rep in reports.items():
        oc_icon_n = "🔴" if rep.overconfidence_normal > 0.1 else "🟡" if rep.overconfidence_normal > 0.03 else "🟢"
        oc_icon_e = "🔴" if rep.overconfidence_extreme > 0.1 else "🟡" if rep.overconfidence_extreme > 0.03 else "🟢"
        deg = rep.accuracy_normal - rep.accuracy_extreme
        deg_icon = "🔴" if deg > 0.15 else "🟡" if deg > 0.07 else "🟢"
        cal_rows.append(
            f"| **{mname}** | "
            f"{rep.accuracy_normal:.1%} | {rep.accuracy_extreme:.1%} | "
            f"{deg_icon} {deg:+.1%} | "
            f"{rep.ece_normal:.3f} | {rep.ece_extreme:.3f} | "
            f"{oc_icon_n} {rep.overconfidence_normal:+.3f} | "
            f"{oc_icon_e} {rep.overconfidence_extreme:+.3f} |"
        )
    cal_table = (
        "| Model | Acc Normal | Acc Extreme | Degradation | ECE Normal | ECE Extreme "
        "| Overconf Normal | Overconf Extreme |\n"
        "|-------|-----------|------------|-------------|-----------|------------|"
        "----------------|------------------|\n"
        + "\n".join(cal_rows)
    )

    explain_md = """
**How Extreme Event Reliability works (from the paper):**

1. **Query classification** — Each query receives an *extremity score* based on:
   - Mean IDF of query tokens (rare/technical terms → higher score)
   - Query length deviation from corpus median
   - Out-of-vocabulary fraction
   Queries above the {pct}th percentile are labelled **extreme**.

2. **Model simulation** — Each strategy generates (confidence, is_correct) per query.
   Strategies are parameterised by their known degradation on extreme inputs:
   - BM25 degrades sharply on rare-term queries (vocabulary gap)
   - Dense retrieval generalises better but can be overconfident on novel phrasing
   - Adaptive hybrid mitigates degradation by per-query alpha tuning

3. **Reliability calibration (ECE)** — Expected Calibration Error measures the gap
   between predicted confidence and actual accuracy across confidence buckets.
   `ECE = Σ (|bucket| / N) × |mean_conf_bucket − frac_correct_bucket|`
   Lower ECE = better calibrated.

4. **Overconfidence** = mean(predicted_confidence) − actual_accuracy.
   Positive = overconfident (model claims higher confidence than it deserves).

**Paper's key finding — applied to RAG:**
AI models that achieve good *average* performance are often poorly calibrated on
**extreme events** (rare queries, OOV terms, adversarial inputs). Specifically:
- Overconfidence on extreme inputs is systematic and larger than on normal inputs
- Models with adaptive mechanisms (adaptive hybrid) show smaller reliability gaps
- ECE on extreme queries ≈ 2–5× ECE on normal queries for non-adaptive methods
""".format(pct=int(extreme_percentile * 100)).strip()

    # ── Reliability diagram plot ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#f9f9f9")
    fig.suptitle("Reliability Diagrams: Normal vs Extreme Queries", fontsize=13)

    cmap = plt.get_cmap("tab10")
    for ax_idx, (split_name, bucket_attr) in enumerate([
        ("Normal queries", "reliability_buckets_normal"),
        ("Extreme queries", "reliability_buckets_extreme"),
    ]):
        ax = axes[ax_idx]
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration", alpha=0.6)
        ax.fill_between([0, 1], [0, 0], [1, 1], alpha=0.04, color="gray")

        for m_idx, (mname, rep) in enumerate(reports.items()):
            buckets = getattr(rep, bucket_attr)
            if not buckets:
                continue
            xs = [b[0] for b in buckets]
            ys = [b[1] for b in buckets]
            ece_val = rep.ece_normal if split_name == "Normal queries" else rep.ece_extreme
            ax.plot(xs, ys, "o-", color=cmap(m_idx), linewidth=1.8, markersize=5,
                    label=f"{mname} (ECE={ece_val:.3f})")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean confidence (in bucket)", fontsize=10)
        ax.set_ylabel("Fraction correct", fontsize=10)
        ax.set_title(split_name, fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return q_table, cal_table, explain_md, fig


EEA_DEMO_QUERIES_LIST = [
    "BM25 Okapi",
    "HNSW graph index approximate nearest neighbour",
    "what is semantic search",
    "ColBERT MaxSim late interaction token embeddings",
    "explain RAG",
    "SPLADE sparse learned vocabulary expansion regularisation",
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
        "| OLR-WAA: Adaptive Drift-Resilient Online Regression | 🔴 8/10 | OLR-WAA |\n"
        "| Bi-Stage Adaptive RAG (Group Recommendation) | 🔴 8/10 | Bi-Stage RAG |\n"
        "| Opinion Maximization with Fairness | 🔴 8/10 | Opinion Maximization |\n"
        "| TopoMAS: LLM-Driven Multi-Agent System (Zhang et al.) | 🔴 8/10 | TopoMAS |\n"
        "| Label Scarcity: Semi-Supervised Pseudo-Labeling | 🔴 8/10 | Label Scarcity |\n"
        "| Extreme Event AI: Reliability Calibration for RAG | 🔴 8/10 | Extreme Event AI |\n"
        "| GrepSeek DCI: Hybrid Search for AI Agents | 🔴 9/10 | GrepSeek DCI |\n"
        "| DeepSeek-Lattice-KG: Knowledge Graph RAG | 🔴 8/10 | KG-RAG |"
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

        # ── Tab 7: Bi-Stage Adaptive RAG ──────────────────────────────────
        with gr.Tab("🎯 Bi-Stage RAG"):
            gr.Markdown(
                "## Bi-Stage Adaptive RAG\n"
                "**Paper: 'Group-Based Recommendation System Using Bi-Stage Adaptive Deep Learning Model' — Score 8/10**\n\n"
                "Adapts group-recommendation's two-stage pipeline to RAG: "
                "**Stage 1** generates per-aspect candidate pools with BM25, "
                "**Stage 2** adaptively reranks using attention weights computed from "
                "aspect–query similarity. No hand-tuning of weights needed — they are "
                "computed on-the-fly from the query content.\n\n"
                "**RAG applicability:** Multi-faceted queries (e.g. 'fast Python RAG for beginners') "
                "decompose into aspects; attention routing ensures the most relevant aspects "
                "dominate the final ranking without static per-query configuration."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    bs_preset_dd = gr.Dropdown(
                        list(BS_PRESETS.keys()),
                        value=list(BS_PRESETS.keys())[0],
                        label="Load preset",
                    )
                    bs_load_btn = gr.Button("Load preset", size="sm")

                    bs_query  = gr.Textbox(label="Main query", lines=1,
                                           placeholder="retrieval augmented generation pipeline")
                    bs_asp1   = gr.Textbox(label="Aspect 1", lines=1,
                                           placeholder="fast efficient low latency")
                    bs_asp2   = gr.Textbox(label="Aspect 2", lines=1,
                                           placeholder="dense vector search embedding")

                with gr.Column(scale=1):
                    bs_topk   = gr.Slider(3, 10, value=5, step=1, label="Top-K results")
                    bs_temp   = gr.Slider(0.1, 2.0, value=0.5, step=0.1,
                                         label="Attention temperature τ",
                                         info="Low → winner-takes-most; High → uniform weights")
                    bs_fixed  = gr.Checkbox(value=False, label="Use manual fixed weights")
                    with gr.Row():
                        bs_fw1 = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                           label="Fixed weight: Aspect 1")
                        bs_fw2 = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                           label="Fixed weight: Aspect 2")

            bs_btn = gr.Button("Run Bi-Stage RAG", variant="primary")

            gr.Markdown("### Stage 1 — Candidate Pool (BM25 per aspect)")
            bs_s1_table = gr.Markdown()

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Stage 2 — Adaptive Re-ranked Results")
                    bs_s2_table = gr.Markdown()
                with gr.Column():
                    bs_attn_md  = gr.Markdown()

            bs_explain = gr.Markdown()

            bs_load_btn.click(
                fn=bs_load_preset,
                inputs=[bs_preset_dd],
                outputs=[bs_query, bs_asp1, bs_asp2],
            )
            bs_btn.click(
                fn=bs_run,
                inputs=[bs_query, bs_asp1, bs_asp2, bs_topk, bs_temp, bs_fixed, bs_fw1, bs_fw2],
                outputs=[bs_s1_table, bs_s2_table, bs_attn_md, bs_explain],
            )
            bs_query.submit(
                fn=bs_run,
                inputs=[bs_query, bs_asp1, bs_asp2, bs_topk, bs_temp, bs_fixed, bs_fw1, bs_fw2],
                outputs=[bs_s1_table, bs_s2_table, bs_attn_md, bs_explain],
            )

            gr.Markdown(
                "---\n"
                "**Key insight from the paper:** In group recommendation, each group member "
                "represents an 'aspect' of collective preference. The bi-stage pipeline first "
                "generates individual candidate sets then uses adaptive attention to weight "
                "each member's contribution — analogous to how multi-aspect queries in RAG "
                "benefit from per-aspect retrieval followed by attention-based fusion.\n\n"
                "**Temperature guide:** τ=0.1 → almost all weight on the most-similar aspect; "
                "τ=1.0 → moderate differentiation; τ=2.0 → near-uniform weights (equivalent to fixed α=0.5)."
            )

        # ── Tab 8: Opinion Maximization with Fairness ─────────────────────
        with gr.Tab("🌐 Opinion Maximization"):
            gr.Markdown(
                "## Opinion Maximization Based on Fairness in Social Networks\n"
                "**Paper: 'Opinion Maximization Based on Fairness in Social Networks' — Score 8/10**\n\n"
                "Selects a set of **seed nodes** in a social graph to maximise the spread of a positive opinion "
                "(e.g., endorsement, product recommendation, or RAG system feedback) while ensuring that "
                "**minority groups** receive proportional coverage — not just the well-connected majority.\n\n"
                "**RAG applicability:** Influence maximisation governs knowledge propagation in graph-RAG systems "
                "(e.g., which documents to prioritise as 'seeds' in a knowledge graph traversal) "
                "and agent orchestration (which agents to activate first to maximise information spread). "
                "Fairness constraints ensure underrepresented topics receive retrieval coverage.\n\n"
                "🔵 **Blue** = majority group nodes &nbsp;&nbsp; 🟠 **Orange** = minority group nodes &nbsp;&nbsp; "
                "**★** = selected seed node"
            )

            with gr.Row():
                with gr.Column(scale=1):
                    om_k       = gr.Slider(1, 8, value=3, step=1,
                                           label="k — number of seeds to select")
                    om_fair    = gr.Slider(0, 5, value=1, step=1,
                                          label="Fairness budget (min minority seeds)")
                    om_sims    = gr.Slider(20, 120, value=60, step=10,
                                          label="Monte-Carlo simulations (higher = more accurate, slower)")
                    om_btn     = gr.Button("Run Optimisation", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown(
                        "### Network structure\n"
                        "- **20 nodes**: 15 majority (blue), 5 minority (orange)\n"
                        "- Intra-group edges are **denser** (homophily)\n"
                        "- Cross-group edges are **sparse** (bridging)\n\n"
                        "### Algorithm\n"
                        "1. **Phase 1 (fairness):** Greedily pick `fairness_budget` seeds from "
                        "minority nodes, maximising marginal spread at each step.\n"
                        "2. **Phase 2 (greedy):** Fill remaining `k − budget` slots from all "
                        "nodes using the standard greedy approximation (≥ 63% of optimum).\n"
                        "3. **Cascade simulation:** Weighted Independent Cascade — each activated "
                        "node spreads opinion to neighbours with probability = edge weight."
                    )

            gr.Markdown("### Comparison: Unconstrained vs Fairness-Constrained")
            om_comparison = gr.Markdown()

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Unconstrained result")
                    om_plain_md = gr.Markdown()
                with gr.Column():
                    gr.Markdown("### Fairness-constrained result")
                    om_fair_md  = gr.Markdown()

            om_plot = gr.Plot(label="Network visualisation")

            om_btn.click(
                fn=om_run,
                inputs=[om_k, om_fair, om_sims],
                outputs=[om_comparison, om_plain_md, om_fair_md, om_plot],
            )

            gr.Markdown(
                "---\n"
                "**Key insight from the paper:** Pure greedy influence maximisation ignores group structure — "
                "it systematically over-selects high-degree nodes in the majority cluster, leaving minority "
                "nodes under-covered. The fairness budget forces at least one minority seed, which "
                "activates the tightly-knit minority cluster and **often increases total spread** "
                "while dramatically closing the coverage gap.\n\n"
                "**RAG analogy:** In a knowledge graph, 'majority nodes' are popular documents linked "
                "by many edges; 'minority nodes' are niche but authoritative sources. "
                "A pure PageRank seed selection would always surface the popular documents — "
                "a fairness budget ensures niche topics receive retrieval coverage."
            )

        # ── Tab 9: TopoMAS ────────────────────────────────────────────
        with gr.Tab("🔬 TopoMAS"):
            gr.Markdown(
                "## TopoMAS: LLM-Driven Topological Materials Multi-Agent System\n"
                "**Paper: Zhang et al. (MGE Advances, 2026) — Score 8/10**\n\n"
                "TopoMAS unifies topological-materials discovery into a coordinated multi-agent pipeline. "
                "A **CoreAgent** decomposes your natural-language query into subtasks and dispatches them "
                "to specialist agents: **KnowledgeAgent** (KG retrieval), **PropertyAgent** (DFT-level analysis), "
                "**SynthesisAgent** (hypothetical candidate generation), and **ValidationAgent** (invariant checking).\n\n"
                "This architecture mirrors production LLM-agent systems (RAG orchestrators, tool-calling chains) "
                "and demonstrates how hierarchical task decomposition improves reliability over monolithic LLM calls."
            )

            with gr.Row():
                with gr.Column(scale=3):
                    topo_query = gr.Textbox(
                        label="Materials query",
                        placeholder="e.g. 'topological insulators for spintronics', 'Bi2Se3 properties', 'suggest new topological material'",
                        lines=2,
                    )
                with gr.Column(scale=1):
                    topo_top_k = gr.Slider(1, 5, value=3, step=1, label="Max KG hits")

            topo_examples = gr.Examples(
                examples=TOPO_DEMO_QUERIES,
                inputs=topo_query,
                label="Example queries",
            )

            topo_btn = gr.Button("Run Multi-Agent Pipeline", variant="primary")

            with gr.Row():
                topo_trace_md = gr.Markdown(label="Agent Execution Trace")

            topo_prop_md = gr.Markdown(label="Property & Validation Table")
            topo_answer_md = gr.Markdown(label="Final Answer")

            topo_btn.click(
                fn=topomas_run,
                inputs=[topo_query, topo_top_k],
                outputs=[topo_trace_md, topo_prop_md, topo_answer_md],
            )

            gr.Markdown(
                "---\n"
                "**Key insight from the paper:** Individual LLMs perform poorly on specialised topological-materials "
                "reasoning when used in isolation. Wrapping them in a multi-agent framework with access to curated "
                "knowledge graphs, troubleshooting databases, and specialised calculation agents **dramatically "
                "improves accuracy and robustness**.\n\n"
                "**RAG / agent analogy:** The same principle applies to production RAG systems — a monolithic "
                "LLM call is fragile; a pipeline with dedicated retrieval, reranking, citation, and validation "
                "agents is more reliable, auditable, and extensible. TopoMAS provides a concrete, "
                "domain-specific existence proof of this architecture."
            )

        # ── Tab 10: Label Scarcity — Semi-Supervised Pseudo-Labeling ─────────
        with gr.Tab("🏷️ Label Scarcity"):
            gr.Markdown(
                "## Label Scarcity: Semi-Supervised Pseudo-Labeling for RAG\n"
                "**Paper: 'Overcoming labelled data scarcity for defect classification "
                "in scanning tunnelling microscopy' — Score 8/10**\n\n"
                "Demonstrates how to **expand a tiny labeled set** into a large pseudo-labeled "
                "training corpus using iterative k-NN assignment in TF-IDF embedding space. "
                "Start with just 3–9 labeled examples per class — the algorithm propagates "
                "labels to unlabeled items by confidence-filtered nearest-neighbor votes, "
                "growing the pool over multiple self-training rounds.\n\n"
                "**RAG applicability:** Query intent classification, retrieval strategy routing "
                "(sparse/dense/hybrid), and embedding fine-tuning data generation — all cases "
                "where you have abundant unlabeled queries but only a handful of annotations."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    ls_preset_dd = gr.Dropdown(
                        list(LS_PRESETS.keys()),
                        value=list(LS_PRESETS.keys())[0],
                        label="Preset scenario",
                        info="Pre-loaded labeled seeds + unlabeled pool",
                    )
                    ls_extra = gr.Textbox(
                        label="Additional unlabeled items (one per line, optional)",
                        lines=4,
                        placeholder="Add your own unlabeled queries here…",
                    )
                with gr.Column(scale=1):
                    ls_threshold = gr.Slider(
                        0.05, 0.9, value=0.25, step=0.05,
                        label="Confidence threshold",
                        info="Min confidence to accept a pseudo-label (lower = more labels, less precise)",
                    )
                    ls_rounds = gr.Slider(
                        1, 5, value=3, step=1,
                        label="Max self-training rounds",
                        info="More rounds = more propagation from early pseudo-labels",
                    )
                    ls_k = gr.Slider(
                        1, 7, value=3, step=1,
                        label="k (nearest neighbors for vote)",
                        info="Higher k = smoother decision boundary, less sensitive to outliers",
                    )

            ls_btn = gr.Button("Run Pseudo-Labeling", variant="primary")

            gr.Markdown("### Seed Labeled Examples")
            ls_seed_table = gr.Markdown()

            gr.Markdown("### Pseudo-Label Results")
            ls_results_table = gr.Markdown()

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Self-Training Round Statistics")
                    ls_stats_md = gr.Markdown()
                with gr.Column():
                    ls_explain_md = gr.Markdown()

            ls_btn.click(
                fn=ls_run,
                inputs=[ls_preset_dd, ls_threshold, ls_rounds, ls_k, ls_extra],
                outputs=[ls_seed_table, ls_results_table, ls_stats_md, ls_explain_md],
            )
            ls_preset_dd.change(
                fn=ls_run,
                inputs=[ls_preset_dd, ls_threshold, ls_rounds, ls_k, ls_extra],
                outputs=[ls_seed_table, ls_results_table, ls_stats_md, ls_explain_md],
            )

            gr.Markdown(
                "---\n"
                "**Key insight from the paper:** Training classifiers with only a few labeled "
                "samples per class is possible when you exploit the geometric structure of the "
                "embedding space — nearby points tend to share the same label. Self-training "
                "iteratively grows the labeled pool, and confidence thresholding prevents "
                "low-quality pseudo-labels from corrupting later rounds.\n\n"
                "**Threshold guide:** 0.05–0.15 → aggressive expansion (many pseudo-labels, "
                "some errors); 0.25–0.40 → balanced; 0.50+ → conservative (high precision, "
                "fewer labels). Start low and increase if you see label noise.\n\n"
                "**RAG workflow:** (1) Label 5–10 queries per intent class manually. "
                "(2) Run pseudo-labeling on 1,000s of unlabeled production queries. "
                "(3) Use the expanded set to fine-tune a sentence transformer or train a "
                "lightweight router — replacing a much costlier full annotation campaign."
            )

        # ── Tab 11: Extreme Event AI — Reliability Calibration ───────────────
        with gr.Tab("🌡️ Extreme Event AI"):
            gr.Markdown(
                "## Extreme Event AI: Reliability Calibration for RAG on Edge-Case Queries\n"
                "**Paper: 'Predictability of precipitation extremes in the Western United States "
                "and the U.S. Gulf Coast using AI weather models' — Score 8/10**\n\n"
                "The paper finds that AI weather models (Aurora, GenCast, Pangu, GraphCast) that "
                "achieve strong *average* performance are systematically **less reliable on "
                "extreme precipitation events** — the tail of the distribution where calibration "
                "and overconfidence matter most.\n\n"
                "**RAG analogy:** Retrieval strategies that look good on average benchmarks can "
                "be poorly calibrated on **extreme queries** — rare vocabulary, very long/short "
                "inputs, out-of-distribution phrasing. This tab quantifies that gap using "
                "**Expected Calibration Error (ECE)** and reliability diagrams split by "
                "normal vs. extreme query sets.\n\n"
                "🟢 **Normal queries** = common vocabulary, average length  \n"
                "🔴 **Extreme queries** = rare terms (high IDF), OOV tokens, length outliers"
            )

            with gr.Row():
                with gr.Column(scale=2):
                    eea_preset_dd = gr.Dropdown(
                        list(EEA_QUERY_PRESETS.keys()) + ["Custom — use text box"],
                        value="Mixed (normal + extreme)",
                        label="Query set",
                        info="Pre-loaded query sets that mix normal and extreme inputs",
                    )
                    eea_custom_txt = gr.Textbox(
                        label="Custom queries (one per line, used when preset = 'Custom')",
                        lines=6,
                        placeholder="Enter one query per line…\n\nBM25 Okapi\nColBERT MaxSim late interaction\nhow does semantic search work",
                    )

                with gr.Column(scale=1):
                    eea_models_dd = gr.CheckboxGroup(
                        list(EEA_MODELS.keys()),
                        value=["BM25 (sparse)", "Dense retrieval", "Adaptive Hybrid (query-IDF α)"],
                        label="Retrieval strategies to compare",
                    )
                    eea_percentile = gr.Slider(
                        0.50, 0.95, value=0.75, step=0.05,
                        label="Extreme percentile threshold",
                        info="Queries above this extremity percentile are labelled 'extreme'",
                    )

            eea_btn = gr.Button("Run Reliability Calibration", variant="primary")

            gr.Markdown("### Query Classification (normal vs extreme)")
            eea_q_table = gr.Markdown()

            gr.Markdown("### Calibration Summary")
            eea_cal_table = gr.Markdown()

            with gr.Row():
                with gr.Column():
                    eea_explain_md = gr.Markdown()
                with gr.Column():
                    eea_plot = gr.Plot(label="Reliability Diagrams")

            eea_btn.click(
                fn=eea_run,
                inputs=[eea_preset_dd, eea_models_dd, eea_percentile, eea_custom_txt],
                outputs=[eea_q_table, eea_cal_table, eea_explain_md, eea_plot],
            )
            eea_preset_dd.change(
                fn=eea_run,
                inputs=[eea_preset_dd, eea_models_dd, eea_percentile, eea_custom_txt],
                outputs=[eea_q_table, eea_cal_table, eea_explain_md, eea_plot],
            )

            gr.Markdown(
                "---\n"
                "**Key insight from the paper:** AI models that perform well on *average* metrics "
                "can be systematically overconfident on extreme events — the tail of the input "
                "distribution where reliability matters most. Reliability diagrams and ECE make "
                "this gap visible and quantifiable.\n\n"
                "**RAG implication:** Standard benchmarks (BEIR, MTEB) evaluate average "
                "performance. Production systems face extreme queries (rare terminology, "
                "adversarial phrasing, OOV tokens) where your retriever's confidence scores "
                "may be poorly calibrated. ECE on an extreme-query split should be part of "
                "every retrieval pipeline evaluation.\n\n"
                "**ECE guide:** 0.00–0.03 → well-calibrated; 0.03–0.08 → moderate; "
                "0.08+ → poorly calibrated. Compare ECE_normal vs ECE_extreme — "
                "a large gap signals that your model's confidence is less trustworthy "
                "precisely when the query is hardest."
            )

        # ------------------------------------------------------------------
        # Tab 12 — GrepSeek DCI
        # ------------------------------------------------------------------
        with gr.Tab("🖥️ GrepSeek DCI"):
            gr.Markdown(
                "## GrepSeek: Direct Corpus Interaction — Hybrid Search for AI Agents\n"
                "**Paper source:** AlphaSignal deep-dive · 2026-05-31 · Score 🔴 9/10\n\n"
                "### Why RAG fails on exact strings\n"
                "Traditional RAG pipelines embed documents into vectors. This blurs exact "
                "identifiers: error codes (`OOM_CODE_0x4F5E`), version constraints "
                "(`transformers>=4.36.0`), alert IDs (`ALERT-INC-2892`). "
                "An AI agent debugging a production incident needs **exact** lookup, not semantic "
                "similarity.\n\n"
                "**Direct Corpus Interaction (DCI)** bypasses embeddings and uses regex/grep "
                "directly over raw document text — the same way a human developer would. "
                "**GrepSeek** extends DCI with RL-based navigation and sharded-parallel execution "
                "for **7.6× throughput** improvement.\n\n"
                "**Hybrid** = semantic retrieval (broad recall) + DCI (precision) fused via RRF:\n"
                "> `score(d) = Σ 1 / (60 + rank_i(d))`\n\n"
                "| Method | nDCG@10 |\n"
                "|--------|---------|\n"
                "| Pure semantic RAG | 0.712 |\n"
                "| Pure BM25 keyword | 0.763 |\n"
                "| GrepSeek DCI only | 0.781 |\n"
                "| **GrepSeek Hybrid (DCI + semantic + RRF)** | **0.847** |"
            )

            with gr.Row():
                with gr.Column(scale=3):
                    dci_query_txt = gr.Textbox(
                        label="Semantic query",
                        placeholder="e.g.  memory error in embedding worker",
                        lines=1,
                    )
                    dci_pattern_txt = gr.Textbox(
                        label="DCI / grep pattern  (regex, optional — defaults to query if empty)",
                        placeholder="e.g.  OOM_CODE_0x4F5E",
                        lines=1,
                    )
                    with gr.Row():
                        dci_mode_radio = gr.Radio(
                            ["Semantic", "DCI / Grep", "Hybrid"],
                            value="Hybrid",
                            label="Search mode",
                        )
                        dci_topk = gr.Slider(1, 12, value=5, step=1, label="Top-K")
                    dci_btn = gr.Button("Search", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("### Corpus preview")
                    corpus_preview_md = "\n".join(
                        f"- **[{d.metadata.get('type','').upper()}]** {d.text[:80]}…"
                        for d in DCI_CORPUS
                    )
                    gr.Markdown(corpus_preview_md)

            gr.Markdown("---")
            with gr.Tabs():
                with gr.Tab("🔀 Active mode result"):
                    dci_main_out = gr.Markdown()
                with gr.Tab("🧠 Semantic only"):
                    dci_sem_out = gr.Markdown()
                with gr.Tab("🖥️ DCI / Grep only"):
                    dci_dci_out = gr.Markdown()

            gr.Examples(
                examples=[
                    ["memory error embedding worker",     "OOM_CODE_0x4F5E",       "Hybrid"],
                    ["rate limit exceeded API",           "ALERT.*C7742|HTTP 429",  "Hybrid"],
                    ["version constraint transformers",   "transformers>=",         "DCI / Grep"],
                    ["how does hybrid search work",       "",                       "Semantic"],
                    ["incident alert vector db timeout",  "ALERT-INC-2892",        "Hybrid"],
                    ["OOM_CODE_0x4F5E",                   "OOM_CODE_0x4F5E",       "DCI / Grep"],
                ],
                inputs=[dci_query_txt, dci_pattern_txt, dci_mode_radio],
                label="Example queries — try semantic-only then DCI-only to see the difference",
            )

            def _dci_run(query: str, pattern: str, mode: str, top_k: int):
                if not query.strip() and not pattern.strip():
                    msg = "_Enter a query or grep pattern._"
                    return msg, msg, msg

                pat = pattern.strip() or query.strip()

                if mode == "Semantic":
                    sem_r = dci_semantic_search(query, top_k=top_k)
                    dci_r = []
                    main_r = sem_r
                elif mode == "DCI / Grep":
                    sem_r = []
                    dci_r = dci_search(pat, top_k=top_k)
                    main_r = dci_r
                else:
                    sem_r = dci_semantic_search(query, top_k=top_k)
                    dci_r = dci_search(pat, top_k=top_k)
                    main_r = dci_hybrid_search(query, pat, top_k=top_k)

                highlight = pat if mode != "Semantic" else query
                return (
                    format_results_md(main_r, highlight),
                    format_results_md(sem_r, query) if sem_r else "_Not used._",
                    format_results_md(dci_r, pat)   if dci_r else "_Not used._",
                )

            dci_btn.click(
                fn=_dci_run,
                inputs=[dci_query_txt, dci_pattern_txt, dci_mode_radio, dci_topk],
                outputs=[dci_main_out, dci_sem_out, dci_dci_out],
            )
            dci_query_txt.submit(
                fn=_dci_run,
                inputs=[dci_query_txt, dci_pattern_txt, dci_mode_radio, dci_topk],
                outputs=[dci_main_out, dci_sem_out, dci_dci_out],
            )

            gr.Markdown(
                "---\n"
                "**Key insight:** Semantic search finds documents _about_ memory errors; "
                "DCI finds documents _containing_ `OOM_CODE_0x4F5E`. Hybrid retrieves both, "
                "ranked by RRF. This is the core of the GrepSeek paper: give agents a "
                "terminal, not just a vector database."
            )

        # ── Tab 13: DeepSeek-Lattice-KG — Knowledge Graph RAG ────────────────
        with gr.Tab("🧠 KG-RAG"):
            gr.Markdown(
                "## DeepSeek-Lattice-KG: Knowledge Graph Augmented RAG\n"
                "**Paper: Shu et al., Xi'an Jiaotong University, MGE Advances 2026 — Score 8/10**\n\n"
                "The paper fine-tunes DeepSeek-R1-14B on domain text, then augments it with a **Neo4j "
                "knowledge graph** built from 50,000 publications. At inference, user queries become "
                "**Cypher graph queries** that retrieve a relevant *subgraph* of entity-relationship "
                "triples — not just similar text chunks. This structured context raises accuracy from "
                "71.7% (base LLM) → 82.9% (fine-tune alone) → **94.8%** (fine-tune + KG).\n\n"
                "**RAG applicability:** The NL → Cypher → subgraph → LLM context pipeline is "
                "domain-agnostic. Below: compare plain BM25 retrieval vs KG-RAG on the same query "
                "over an AI engineering knowledge graph."
            )

            with gr.Row():
                with gr.Column(scale=3):
                    kg_query = gr.Textbox(
                        label="Query",
                        placeholder="e.g. 'how does KG-RAG reduce hallucination?' or 'explain hybrid search'",
                        lines=2,
                    )
                with gr.Column(scale=1):
                    kg_hops = gr.Slider(1, 2, value=1, step=1, label="Graph hops")

            gr.Examples(
                examples=[
                    ["how does KG-RAG reduce hallucination?"],
                    ["explain hybrid search and its components"],
                    ["what is the difference between fine-tuning and RAG?"],
                    ["how are embedding models used in dense retrieval?"],
                    ["why does hallucination happen and how to fix it?"],
                ],
                inputs=kg_query,
                label="Example queries",
            )

            kg_btn = gr.Button("Run KG-RAG vs BM25", variant="primary")

            with gr.Row():
                with gr.Column():
                    kg_entities_md = gr.Markdown(label="Extracted Entities + Cypher Queries")
                with gr.Column():
                    kg_bm25_md = gr.Markdown(label="Plain BM25 Results")

            kg_context_md = gr.Markdown(label="KG-RAG Retrieved Subgraph Context")
            kg_graph_img = gr.Plot(label="Subgraph Visualization")

            def _kg_run(query: str, hops: int):
                if not query.strip():
                    return "_Enter a query._", "_Enter a query._", "_Enter a query._", None

                # KG-RAG pipeline
                entities, cyphers, sub_nodes, sub_edges, context = kg_rag_search(query, hops=hops)

                # Entity + Cypher display
                ent_lines = ["### Extracted Entities\n"]
                for e in entities:
                    ent_lines.append(f"- **{e}**")
                ent_lines.append("\n### Generated Cypher Queries\n")
                for c in cyphers:
                    ent_lines.append(f"```cypher\n{c}\n```\n")
                ent_md = "\n".join(ent_lines)

                # BM25 baseline
                bm25_results = bm25_search(query, top_k=4)
                bm25_lines = ["### BM25 Results (plain text retrieval)\n"]
                for title, text, score in bm25_results:
                    bm25_lines.append(f"**{title}** (score: {score:.3f})\n> {text[:120]}…\n")
                bm25_md = "\n".join(bm25_lines)

                # Subgraph visualization
                fig = _draw_subgraph(entities, sub_nodes, sub_edges)

                return ent_md, bm25_md, context, fig

            def _draw_subgraph(
                anchor_entities: list[str],
                sub_nodes: set[str],
                sub_edges: list[KGEdge],
            ):
                fig, ax = plt.subplots(figsize=(10, 7))
                ax.set_facecolor("#0f1117")
                fig.patch.set_facecolor("#0f1117")

                if not sub_nodes:
                    ax.text(0.5, 0.5, "No subgraph to display", ha="center", va="center",
                            color="white", transform=ax.transAxes)
                    return fig

                nodes = list(sub_nodes)
                n = len(nodes)
                idx = {name: i for i, name in enumerate(nodes)}

                # circular layout
                angles = [2 * np.pi * i / n for i in range(n)]
                xs = [np.cos(a) for a in angles]
                ys = [np.sin(a) for a in angles]

                # draw edges
                for e in sub_edges:
                    if e.src in idx and e.dst in idx:
                        x0, y0 = xs[idx[e.src]], ys[idx[e.src]]
                        x1, y1 = xs[idx[e.dst]], ys[idx[e.dst]]
                        ax.annotate(
                            "", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="->", color="#555", lw=1.0,
                                            connectionstyle="arc3,rad=0.1"),
                        )
                        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                        ax.text(mx, my, e.rel, fontsize=6, color="#aaa", ha="center",
                                va="center", style="italic")

                # draw nodes
                for name in nodes:
                    i = idx[name]
                    color = "#e74c3c" if name in anchor_entities else "#3498db"
                    ax.scatter(xs[i], ys[i], s=800, c=color, zorder=3, alpha=0.9)
                    ax.text(xs[i], ys[i] + 0.12, name, fontsize=8, ha="center",
                            va="bottom", color="white", fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a2e", ec="none", alpha=0.7))

                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.5, 1.5)
                ax.axis("off")
                ax.set_title(
                    "Retrieved Subgraph  (🔴 = query entity · 🔵 = neighbor)",
                    color="white", fontsize=11, pad=12,
                )
                fig.tight_layout()
                return fig

            kg_btn.click(
                fn=_kg_run,
                inputs=[kg_query, kg_hops],
                outputs=[kg_entities_md, kg_bm25_md, kg_context_md, kg_graph_img],
            )
            kg_query.submit(
                fn=_kg_run,
                inputs=[kg_query, kg_hops],
                outputs=[kg_entities_md, kg_bm25_md, kg_context_md, kg_graph_img],
            )

            gr.Markdown(
                "---\n"
                "**Key insight from the paper:** Plain RAG retrieves similar *documents*; "
                "KG-RAG retrieves *entity relationships* — it knows that `KG-RAG --reduces--> Hallucination` "
                "because that edge exists in the graph, not just because the words appear near each other. "
                "The 71.7% → 94.8% jump shows that structured relational context adds information "
                "that dense-vector similarity alone cannot recover.\n\n"
                "**Pattern:** NL query → entity extraction → Cypher → subgraph → structured context → LLM. "
                "This pipeline is domain-agnostic: swap the lattice-materials KG for your domain's KG "
                "(legal entities, product catalog, code API graph) and the architecture is identical."
            )

if __name__ == "__main__":
    demo.launch()
