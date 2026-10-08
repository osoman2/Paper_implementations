"""
EmbeddingGemma 2: Quantized Multimodal Embeddings (text + audio + video)
Source: AlphaSignal Signal · 2026-10-08 · 11,470 downloads
Score: 9/10 — Drop-in multimodal embedding model for RAG pipelines
"""

import math
import random

# ── Modality overview ─────────────────────────────────────────────────────────

MODALITY_OVERVIEW = [
    {
        "modality": "Text",
        "icon": "📝",
        "description": "Dense semantic embeddings for documents, queries, and passages.",
        "dim": 2048,
        "max_tokens": 8192,
        "use_cases": ["document RAG", "semantic search", "query routing", "dedup"],
        "latency_ms": 12,
        "model_size_mb": 180,
    },
    {
        "modality": "Audio",
        "icon": "🎙️",
        "description": "Spectral + semantic embeddings for speech and sound clips.",
        "dim": 2048,
        "max_tokens": None,
        "max_seconds": 60,
        "use_cases": ["voice Q&A", "podcast retrieval", "meeting search", "audio dedup"],
        "latency_ms": 35,
        "model_size_mb": 210,
    },
    {
        "modality": "Video",
        "icon": "🎬",
        "description": "Frame-level + temporal embeddings for video content.",
        "dim": 2048,
        "max_tokens": None,
        "max_seconds": 120,
        "use_cases": ["video search", "clip retrieval", "visual Q&A", "scene indexing"],
        "latency_ms": 85,
        "model_size_mb": 260,
    },
]

# ── Benchmark comparison data ─────────────────────────────────────────────────

BENCHMARK_DATA = [
    # Text retrieval benchmarks
    {
        "benchmark": "MTEB (Avg)",
        "modality": "Text",
        "embedding_gemma2_q4": 68.1,
        "embedding_gemma2_fp16": 69.4,
        "text_embedding_3_small": 62.3,
        "voyage_2": 70.1,
        "e5_large": 64.8,
        "notes": "Quantized (Q4) within 1.3 pts of FP16; 5.8 pts above text-embedding-3-small",
    },
    {
        "benchmark": "BEIR (Avg NDCG@10)",
        "modality": "Text",
        "embedding_gemma2_q4": 54.2,
        "embedding_gemma2_fp16": 55.0,
        "text_embedding_3_small": 51.6,
        "voyage_2": 56.2,
        "e5_large": 52.9,
        "notes": "Competitive on zero-shot domain transfer",
    },
    # Audio retrieval benchmarks
    {
        "benchmark": "AudioCaps Recall@10",
        "modality": "Audio",
        "embedding_gemma2_q4": 72.4,
        "embedding_gemma2_fp16": 73.8,
        "clap_laion": 68.2,
        "wav2clip": 61.5,
        "notes": "4.2 pts above CLAP-LAION on cross-modal text→audio retrieval",
    },
    # Video retrieval benchmarks
    {
        "benchmark": "MSR-VTT Recall@10",
        "modality": "Video",
        "embedding_gemma2_q4": 64.7,
        "embedding_gemma2_fp16": 66.1,
        "clip_vit_l14": 60.3,
        "internvideo2": 67.9,
        "notes": "Within 1.8 pts of InternVideo2 at 3x lower inference cost",
    },
    # Cross-modal retrieval
    {
        "benchmark": "X-Modal Recall@10 (any→any)",
        "modality": "Cross",
        "embedding_gemma2_q4": 58.3,
        "embedding_gemma2_fp16": 59.7,
        "imagebind": 54.1,
        "notes": "Shared embedding space: audio, text, and video directly comparable",
    },
]

# ── Simulated corpus for live demo ────────────────────────────────────────────

_CORPUS = {
    "text": [
        ("T001", "RAG pipeline tutorial: chunking strategies and embedding selection for production"),
        ("T002", "Hybrid search with BM25 and dense retrieval for improved recall"),
        ("T003", "Agent memory management using Redis and conversation summarisation"),
        ("T004", "Speculative decoding techniques to reduce LLM inference latency by 3x"),
        ("T005", "Multimodal document parsing: extracting tables, charts, and images for RAG"),
        ("T006", "LLM evaluation metrics: RAGAS, ROUGE, and semantic similarity scoring"),
    ],
    "audio": [
        ("A001", "Podcast episode: deep dive on hybrid search tuning for production RAG"),
        ("A002", "Conference talk: reducing hallucinations with retrieval-augmented generation"),
        ("A003", "Webinar recording: agent orchestration patterns for enterprise deployments"),
        ("A004", "Interview: Anthropic researchers on Claude's tool-use and context management"),
        ("A005", "Tutorial audio: implementing BM25 and vector search in Weaviate"),
    ],
    "video": [
        ("V001", "Live coding session: building a RAG pipeline from scratch with LangChain"),
        ("V002", "Workshop recording: fine-tuning embedding models on domain-specific data"),
        ("V003", "Demo video: real-time agent orchestration with streaming responses"),
        ("V004", "Conference keynote: the future of retrieval-augmented generation"),
        ("V005", "Tutorial: multimodal RAG with text, audio, and image retrieval"),
    ],
}

_MODALITY_LABELS = {"text": "📝 Text", "audio": "🎙️ Audio", "video": "🎬 Video"}


def _cosine_sim_mock(query: str, doc_text: str) -> float:
    """Deterministic mock similarity: keyword overlap heuristic."""
    q_words = set(query.lower().split())
    d_words = set(doc_text.lower().split())
    overlap = len(q_words & d_words)
    base = overlap / (math.sqrt(len(q_words)) * math.sqrt(max(len(d_words), 1)) + 1e-9)
    # Add small per-doc noise seeded by doc text for reproducibility
    seed = sum(ord(c) for c in doc_text[:20])
    rng = random.Random(seed)
    noise = rng.uniform(-0.05, 0.05)
    return max(0.0, min(1.0, base * 4 + 0.3 + noise))


def run_similarity_demo(query: str, modalities: list[str], top_k: int = 3) -> str:
    """Simulate cross-modal similarity search across selected modalities."""
    if not query.strip():
        return "_Please enter a query._"

    all_results = []
    for mod in modalities:
        docs = _CORPUS.get(mod, [])
        for doc_id, doc_text in docs:
            score = _cosine_sim_mock(query, doc_text)
            all_results.append((score, mod, doc_id, doc_text))

    all_results.sort(reverse=True)
    top = all_results[:top_k]

    lines = [f"### Top {top_k} results across {', '.join(modalities)} for: *\"{query}\"*\n"]
    for rank, (score, mod, doc_id, doc_text) in enumerate(top, 1):
        badge = _MODALITY_LABELS[mod]
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        lines.append(
            f"**#{rank}** {badge} · `{doc_id}` · **{score:.3f}**\n"
            f"`[{bar}]`\n"
            f"> {doc_text}\n"
        )

    latencies = {"text": 12, "audio": 35, "video": 85}
    total_lat = sum(latencies.get(m, 0) for m in modalities)
    lines.append(f"\n---\n*Estimated embed latency: {total_lat}ms · Shared 2048-dim space*")
    return "\n".join(lines)


def run_rag_retrieval_demo(question: str, context_modalities: list[str]) -> str:
    """Simulate a multimodal RAG retrieval + grounded answer."""
    if not question.strip():
        return "_Please enter a question._"

    retrieved = []
    for mod in context_modalities:
        docs = _CORPUS.get(mod, [])
        scored = [((_cosine_sim_mock(question, t)), i, t) for i, t in docs]
        scored.sort(reverse=True)
        for score, doc_id, doc_text in scored[:2]:
            retrieved.append((score, mod, doc_id, doc_text))

    retrieved.sort(reverse=True)
    top_ctx = retrieved[:4]

    lines = [f"### Multimodal RAG Answer for: *\"{question}\"*\n"]
    lines.append("**Retrieved context:**\n")
    for score, mod, doc_id, doc_text in top_ctx:
        badge = _MODALITY_LABELS[mod]
        lines.append(f"- {badge} `{doc_id}` ({score:.2f}): {doc_text}")

    lines.append("\n**Grounded answer (simulated):**\n")
    if top_ctx:
        top_doc = top_ctx[0][3]
        lines.append(
            f"> Based on retrieved sources, particularly *{top_doc[:60]}...*, "
            f"the answer draws on content from {len(top_ctx)} cross-modal sources "
            f"({', '.join(set(m for _, m, _, _ in top_ctx))})."
        )
    else:
        lines.append("> No relevant context found across selected modalities.")

    lines.append(
        f"\n---\n*Modalities searched: {', '.join(context_modalities)} · "
        f"Embedding model: EmbeddingGemma 2 Q4 · Shared 2048-dim space*"
    )
    return "\n".join(lines)


# ── Implementation patterns ────────────────────────────────────────────────────

IMPL_PATTERNS = [
    {
        "name": "1. Drop-In Text RAG Replacement",
        "description": (
            "Replace your existing embedding model with EmbeddingGemma 2 for better "
            "MTEB performance at lower cost via Q4 quantization. Compatible with any "
            "vector store that accepts 2048-dim float32 vectors."
        ),
        "code": """\
from unsloth import EmbeddingGemma2  # pip install unsloth

model = EmbeddingGemma2.from_pretrained(
    "unsloth/EmbeddingGemma2-2B-Q4_K_M",
    quantization="q4_k_m",
    device="cpu",          # or "cuda"
)

def embed_texts(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, batch_size=32, normalize=True).tolist()

# Drop-in for OpenAI / Cohere embeddings — same interface
embeddings = embed_texts(["RAG pipeline best practices"])
""",
    },
    {
        "name": "2. Multimodal Cross-Modal Retrieval",
        "description": (
            "Embed text, audio, and video into the same 2048-dim space. "
            "A text query retrieves the most relevant audio clips and video scenes "
            "alongside text documents — no separate models needed."
        ),
        "code": """\
from unsloth import EmbeddingGemma2

model = EmbeddingGemma2.from_pretrained("unsloth/EmbeddingGemma2-2B-Q4_K_M")

# Embed different modalities — same space
text_emb   = model.encode_text("How to tune BM25 alpha for hybrid search")
audio_emb  = model.encode_audio("path/to/podcast_episode.mp3")
video_emb  = model.encode_video("path/to/tutorial.mp4")

# All embeddings are directly comparable via cosine similarity
import numpy as np
sim_text_audio = np.dot(text_emb, audio_emb)  # cross-modal similarity
sim_text_video = np.dot(text_emb, video_emb)

print(f"Text↔Audio: {sim_text_audio:.3f}, Text↔Video: {sim_text_video:.3f}")
""",
    },
    {
        "name": "3. Multimodal RAG Pipeline",
        "description": (
            "A production-ready multimodal RAG pipeline that indexes and retrieves "
            "across text documents, audio recordings, and video clips using a single "
            "embedding model and a shared vector index."
        ),
        "code": """\
from unsloth import EmbeddingGemma2
import numpy as np

model = EmbeddingGemma2.from_pretrained("unsloth/EmbeddingGemma2-2B-Q4_K_M")

class MultimodalRAG:
    def __init__(self):
        self.index = []  # (embedding, modality, content_id, metadata)

    def add_text(self, text: str, doc_id: str):
        emb = model.encode_text(text, normalize=True)
        self.index.append((emb, "text", doc_id, text[:100]))

    def add_audio(self, path: str, clip_id: str):
        emb = model.encode_audio(path, normalize=True)
        self.index.append((emb, "audio", clip_id, path))

    def add_video(self, path: str, clip_id: str):
        emb = model.encode_video(path, normalize=True)
        self.index.append((emb, "video", clip_id, path))

    def search(self, query: str, top_k: int = 5):
        q_emb = model.encode_text(query, normalize=True)
        scored = [
            (float(np.dot(q_emb, item[0])), item[1], item[2], item[3])
            for item in self.index
        ]
        return sorted(scored, reverse=True)[:top_k]

# Usage
rag = MultimodalRAG()
rag.add_text("RAG pipeline chunking strategies...", "doc_001")
rag.add_audio("episode_42.mp3", "pod_042")
rag.add_video("tutorial_rag.mp4", "vid_003")

results = rag.search("How to tune chunking for RAG?")
for score, modality, content_id, preview in results:
    print(f"[{score:.3f}] {modality}: {content_id} — {preview}")
""",
    },
]
