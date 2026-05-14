"""
Query-Adaptive Hybrid Search — Gradio demo.

Implements the vstash IDF-adaptive alpha approach:
  alpha = f(mean_IDF(query tokens))  →  blends BM25 + dense scores per query.

Run:
  python src/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from retriever import Document, SparseRetriever, DenseRetriever
from query_analyzer import QueryAnalyzer
from fusion import score_fusion, rrf_fusion


# ---------------------------------------------------------------------------
# Sample corpus — diverse enough to exercise both retrieval modes
# ---------------------------------------------------------------------------
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

print("Loading models (first run downloads ~90 MB)…")
sparse_ret = SparseRetriever(CORPUS)
dense_ret  = DenseRetriever(CORPUS)
analyzer   = QueryAnalyzer(corpus_texts)
print("Models ready.")


# ---------------------------------------------------------------------------
# Search logic
# ---------------------------------------------------------------------------

def search(query: str, fusion_method: str, top_k: int, force_alpha: float, auto_alpha: bool):
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

    # --- format results table ---
    rows = []
    for rank, (doc_id, score) in enumerate(results, 1):
        doc = doc_map[doc_id]
        rows.append(f"| {rank} | {doc_id} | {score:.4f} | {doc.text} |")

    table = (
        "| Rank | ID | Score | Document |\n"
        "|------|----|-------|----------|\n"
        + "\n".join(rows)
    )

    # --- alpha explanation ---
    alpha_info = (
        f"**Alpha (sparse weight): {alpha}**  \n"
        f"Query type: *{q_type}*  \n\n"
        f"{'🔡 High IDF → sparse (BM25) preferred' if alpha >= 0.65 else '🔍 Low IDF → dense (semantic) preferred' if alpha <= 0.35 else '⚖️ Balanced — both retrievers contribute equally'}"
    )

    # --- compare fixed vs adaptive ---
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
# Gradio UI
# ---------------------------------------------------------------------------

demo_queries = [
    "BM25 Okapi",
    "how do transformers handle long sequences",
    "Mixtral 8x7B sparse mixture of experts",
    "retrieval augmented generation knowledge",
    "Karpukhin dense passage",
]

with gr.Blocks(title="Query-Adaptive Hybrid Search", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Query-Adaptive Hybrid Search\n"
        "**POC based on vstash (arXiv:2604.15484) & DAT (arXiv:2503.23013)**\n\n"
        "Each query gets its own optimal `alpha` weight between BM25 (sparse) and "
        "semantic search (dense), computed from the query's mean IDF. "
        "High-IDF / rare-term queries lean sparse; common-word / conceptual queries lean dense."
    )

    with gr.Row():
        with gr.Column(scale=3):
            query_box = gr.Textbox(label="Query", placeholder="Try a keyword query or a semantic question…", lines=1)
            gr.Examples(demo_queries, inputs=query_box, label="Example queries")

        with gr.Column(scale=1):
            fusion_dd  = gr.Dropdown(["Score fusion", "RRF fusion"], value="Score fusion", label="Fusion method")
            top_k_sl   = gr.Slider(1, 10, value=5, step=1, label="Top-K results")
            auto_cb    = gr.Checkbox(value=True, label="Auto alpha (IDF-adaptive)")
            alpha_sl   = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Manual alpha (0=dense, 1=sparse)",
                                   interactive=True)

    search_btn = gr.Button("Search", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Results")
            results_md = gr.Markdown()
        with gr.Column():
            gr.Markdown("### Alpha Analysis")
            alpha_md = gr.Markdown()
            gr.Markdown("### Fixed vs Adaptive Comparison")
            compare_md = gr.Markdown()

    search_btn.click(
        fn=search,
        inputs=[query_box, fusion_dd, top_k_sl, alpha_sl, auto_cb],
        outputs=[results_md, alpha_md, compare_md],
    )
    query_box.submit(
        fn=search,
        inputs=[query_box, fusion_dd, top_k_sl, alpha_sl, auto_cb],
        outputs=[results_md, alpha_md, compare_md],
    )

if __name__ == "__main__":
    demo.launch()
