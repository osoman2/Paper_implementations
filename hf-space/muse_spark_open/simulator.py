"""
Muse Spark Open — MoE Architecture Cards, Benchmarks, Cost Calculator, Implementation Patterns
Source: AlphaSignal Top News · 2026-10-16 · news@alphasignal.ai
Score: 8/10
"""

import math

# ---------------------------------------------------------------------------
# Architecture cards
# ---------------------------------------------------------------------------
ARCHITECTURE_CARDS = [
    {
        "icon": "🧠",
        "name": "Mixture-of-Experts (MoE) Routing",
        "summary": "816B total parameters, only 42B active per request via sparse gating",
        "description": (
            "Muse Spark Open uses a **Mixture-of-Experts** architecture: 256 specialist "
            "sub-networks (experts) are stored in weights, but for any given input token "
            "only 8 experts are activated by a lightweight gating network (top-8 routing).\n\n"
            "**Why this matters for inference cost:**\n"
            "- Dense 816B model would cost ~816B FLOPs per forward pass\n"
            "- Muse Spark Open costs ~42B FLOPs per forward pass (8/256 × 816B ≈ 42B active)\n"
            "- Quality approaches a dense 816B model while compute matches a 42B dense model\n\n"
            "**Gating mechanism:** A small router (linear layer + softmax) assigns each token "
            "to its top-8 experts. The router is trained jointly with the experts so routing "
            "decisions are task-adaptive, not random.\n\n"
            "**For RAG/agent pipelines:** You can reason about *cost as if the model were 42B* "
            "while retaining the *quality of a model trained on 816B capacity*. This changes "
            "how you budget token throughput vs. quality in production pipelines."
        ),
    },
    {
        "icon": "📏",
        "name": "1M Token Context Window",
        "summary": "Entire codebases, long documents, or multi-session histories in a single call",
        "description": (
            "The 1 million token context window (≈ 750k words) changes the RAG design space:\n\n"
            "| Context Size | Typical RAG Strategy |\n"
            "|---|---|\n"
            "| 4k–32k tokens | Chunking + vector search required |\n"
            "| 128k tokens | Few large docs fit; hybrid RAG still useful |\n"
            "| **1M tokens** | **Entire codebase, book, or multi-day log in one call** |\n\n"
            "**Architectural shift:** At 1M context you can skip the retrieval step for many "
            "workloads and pass the entire corpus to the model. This eliminates chunking errors, "
            "retrieval misses, and context assembly logic.\n\n"
            "**Trade-off:** Long contexts still incur quadratic attention cost. For workloads "
            "with many short queries over a static corpus, pre-built vector indexes remain "
            "faster and cheaper. Use 1M context for: compliance review, code understanding, "
            "document comparison, long-horizon agentic tasks."
        ),
    },
    {
        "icon": "🖼️",
        "name": "Native Multimodal Input",
        "summary": "Single model handles text, images, video, and audio — no preprocessing pipeline",
        "description": (
            "Unlike many LLMs that require separate vision encoders wired via adapters, "
            "Muse Spark Open natively encodes text, images, video frames, and audio in a "
            "unified token space.\n\n"
            "**Pipeline simplification:**\n"
            "- Remove: separate OCR service → structured text → LLM input\n"
            "- Remove: vision encoder + projection layer wiring\n"
            "- Remove: audio transcription step before LLM handoff\n"
            "- Keep: one API call with mixed-modality inputs\n\n"
            "**For agent pipelines:** Tool outputs (screenshots, audio recordings, video "
            "frames) can be passed directly as observation tokens without preprocessing. "
            "This simplifies computer-use agents and multimodal RAG pipelines."
        ),
    },
    {
        "icon": "💻",
        "name": "Local Inference via MLX on Apple Silicon",
        "summary": "Two M5 Ultra Mac Studios run the full 816B model — no cloud required",
        "description": (
            "The MLX framework (Apple's array framework for Apple Silicon) enables multi-GPU "
            "tensor parallelism across Mac Studios over Thunderbolt.\n\n"
            "**Hardware requirements:**\n"
            "- 2× Apple M5 Ultra Mac Studio (192 GB unified memory each = 384 GB total)\n"
            "- FP8/INT4 quantization to fit 816B params in 384 GB\n"
            "- Thunderbolt 4 interconnect for tensor parallelism\n\n"
            "**Why this matters for production:**\n"
            "- Air-gapped / on-premise deployments of a frontier-quality model\n"
            "- HIPAA/GDPR compliance use cases where cloud is not permitted\n"
            "- One-time hardware cost vs. ongoing API per-token spend\n\n"
            "**API availability:** Also served on Groq (ultra-low latency), Together AI, "
            "and OpenRouter (unified gateway) for cloud deployments."
        ),
    },
]

# ---------------------------------------------------------------------------
# Benchmark table
# ---------------------------------------------------------------------------
BENCHMARK_ROWS = [
    {
        "model": "Muse Spark Open (816B MoE)",
        "active_params": "42B",
        "context": "1M",
        "coding_score": "68.4%",
        "multimodal": "✅",
        "local": "✅",
        "cost_per_1m_input": "$0.18",
    },
    {
        "model": "Llama 4 Scout (109B MoE)",
        "active_params": "17B",
        "context": "128K",
        "coding_score": "54.2%",
        "multimodal": "✅",
        "local": "✅",
        "cost_per_1m_input": "$0.10",
    },
    {
        "model": "Mistral Large 4 (1T MoE)",
        "active_params": "~100B",
        "context": "256K",
        "coding_score": "61.7%",
        "multimodal": "✅",
        "local": "❌",
        "cost_per_1m_input": "$0.90",
    },
    {
        "model": "Gemini 4 Argon",
        "active_params": "Dense",
        "context": "1M",
        "coding_score": "72.1%",
        "multimodal": "✅",
        "local": "❌",
        "cost_per_1m_input": "$1.25",
    },
    {
        "model": "Claude Sonnet 5.5",
        "active_params": "Dense",
        "context": "200K",
        "coding_score": "69.8%",
        "multimodal": "✅",
        "local": "❌",
        "cost_per_1m_input": "$3.00",
    },
]


def benchmark_table_html() -> str:
    rows_html = ""
    for r in BENCHMARK_ROWS:
        highlight = ' style="background:#1a3a1a;"' if "Muse Spark Open" in r["model"] else ""
        rows_html += (
            f"<tr{highlight}>"
            f"<td><b>{r['model']}</b></td>"
            f"<td>{r['active_params']}</td>"
            f"<td>{r['context']}</td>"
            f"<td>{r['coding_score']}</td>"
            f"<td>{r['multimodal']}</td>"
            f"<td>{r['local']}</td>"
            f"<td>{r['cost_per_1m_input']}</td>"
            f"</tr>"
        )
    return f"""
<table style="width:100%;border-collapse:collapse;font-size:0.9em;">
  <thead>
    <tr style="background:#2a2a3a;">
      <th style="padding:8px;text-align:left;">Model</th>
      <th style="padding:8px;text-align:center;">Active Params</th>
      <th style="padding:8px;text-align:center;">Context</th>
      <th style="padding:8px;text-align:center;">Coding Score</th>
      <th style="padding:8px;text-align:center;">Multimodal</th>
      <th style="padding:8px;text-align:center;">Local</th>
      <th style="padding:8px;text-align:center;">Cost/1M Input</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
"""


# ---------------------------------------------------------------------------
# Cost / context calculator
# ---------------------------------------------------------------------------

def compute_rag_cost(
    num_documents: int,
    avg_doc_tokens: int,
    queries_per_day: int,
    use_full_context: bool,
    model_cost_per_1m: float,
) -> str:
    """Compare RAG with retrieval vs. full-context stuffing cost."""
    total_corpus_tokens = num_documents * avg_doc_tokens

    if use_full_context:
        # Full context: pass entire corpus on every query
        tokens_per_query = total_corpus_tokens
        strategy = "Full-context (no retrieval)"
        note = (
            "⚠️ This is only feasible when total corpus fits in context window. "
            f"Corpus = {total_corpus_tokens:,} tokens."
        )
    else:
        # Retrieval: assume top-5 chunks of 512 tokens each + query
        retrieved_tokens = 5 * 512
        tokens_per_query = retrieved_tokens + 200  # +200 for query + system prompt
        strategy = "RAG with retrieval (top-5 × 512 tokens)"
        note = "Vector search cost not included."

    daily_input_tokens = tokens_per_query * queries_per_day
    daily_cost = (daily_input_tokens / 1_000_000) * model_cost_per_1m
    monthly_cost = daily_cost * 30

    result = (
        f"## Strategy: {strategy}\n\n"
        f"| Metric | Value |\n"
        f"|---|---|\n"
        f"| Corpus size | {total_corpus_tokens:,} tokens ({total_corpus_tokens/1_000_000:.2f}M) |\n"
        f"| Tokens per query | {tokens_per_query:,} |\n"
        f"| Queries per day | {queries_per_day:,} |\n"
        f"| Daily input tokens | {daily_input_tokens:,} |\n"
        f"| Daily cost @ ${model_cost_per_1m}/1M | **${daily_cost:.2f}** |\n"
        f"| Monthly cost | **${monthly_cost:.2f}** |\n\n"
        f"_{note}_\n\n"
    )

    if not use_full_context:
        # Also show what full context would cost for comparison
        fc_tokens_per_query = total_corpus_tokens
        fc_daily = (fc_tokens_per_query * queries_per_day / 1_000_000) * model_cost_per_1m
        fits = total_corpus_tokens <= 1_000_000
        result += (
            f"**Full-context alternative:** ${fc_daily:.2f}/day "
            f"({'✅ fits in 1M context' if fits else '❌ exceeds 1M context — retrieval required'})\n"
        )
        if fits and fc_daily < daily_cost * 2:
            result += (
                "\n💡 **Recommendation:** For this corpus size the full-context approach "
                "costs only " + f"{fc_daily/daily_cost:.1f}×" + " more but eliminates all "
                "retrieval complexity. Consider using Muse Spark Open's 1M context directly."
            )

    return result


# ---------------------------------------------------------------------------
# Implementation patterns
# ---------------------------------------------------------------------------
IMPL_PATTERNS = [
    {
        "name": "1. Cloud API via Groq (ultra-low latency)",
        "description": (
            "Groq's LPU inference hardware achieves the lowest latency for Muse Spark Open. "
            "Best for interactive agents and real-time applications."
        ),
        "code": """\
from groq import Groq

client = Groq()  # GROQ_API_KEY from env

def query_muse_spark(prompt: str, context_docs: list[str] | None = None) -> str:
    messages = []
    if context_docs:
        context_str = "\\n\\n---\\n\\n".join(context_docs)
        messages.append({
            "role": "user",
            "content": (
                f"Context:\\n{context_str}\\n\\n"
                f"Question: {prompt}"
            ),
        })
    else:
        messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="meta/muse-spark-open",  # check Groq model hub for exact ID
        messages=messages,
        temperature=0.1,
        max_tokens=4096,
    )
    return response.choices[0].message.content
""",
    },
    {
        "name": "2. Local MLX inference on Apple Silicon",
        "description": (
            "Run the full 816B model locally on two M5 Ultra Mac Studios via MLX tensor "
            "parallelism. Requires the mlx-lm package and model weights from Hugging Face."
        ),
        "code": """\
# pip install mlx-lm huggingface-hub
from mlx_lm import load, generate

# Load across two Mac Studios via Thunderbolt (MLX handles tensor parallelism)
model, tokenizer = load(
    "meta/muse-spark-open",    # HF repo ID
    # quantize to INT4 to fit in 384 GB unified memory
    tokenizer_config={"trust_remote_code": True},
)

def query_local(prompt: str) -> str:
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=2048,
        verbose=False,
    )
    return response
""",
    },
    {
        "name": "3. Long-context RAG: skip retrieval, pass full corpus",
        "description": (
            "When your corpus fits within 1M tokens, pass it directly instead of building "
            "a retrieval index. Removes chunking errors and retrieval misses at the cost "
            "of higher per-query token spend."
        ),
        "code": """\
import openai  # OpenRouter is OpenAI-compatible

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="OPENROUTER_API_KEY",  # from env
)

def full_context_rag(documents: list[str], question: str) -> str:
    \"\"\"
    Pass all documents in context — no chunking, no vector search.
    Use when sum(len(doc) for doc in documents) / 4 < 1_000_000 tokens.
    \"\"\"
    corpus = "\\n\\n===\\n\\n".join(documents)
    response = client.chat.completions.create(
        model="meta/muse-spark-open",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer based only on the "
                    "provided documents. Quote the relevant passage."
                ),
            },
            {
                "role": "user",
                "content": f"Documents:\\n{corpus}\\n\\nQuestion: {question}",
            },
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    return response.choices[0].message.content
""",
    },
]
