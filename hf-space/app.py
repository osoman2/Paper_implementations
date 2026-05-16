"""
Paper POCs — multi-tab Gradio demo.

Tabs:
  1. Query-Adaptive Hybrid Search (DAT / vstash) — score 10/10
  2. CAD Score: Context-Aware Semantic Diversity — score 9/10
  3. Dynamic Sub-Sequence Warping (DSSW) — score 9/10
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
        "| Dynamic Sub-Sequence Warping (DSSW) | 🔴 9/10 | DSSW |"
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

if __name__ == "__main__":
    demo.launch()
