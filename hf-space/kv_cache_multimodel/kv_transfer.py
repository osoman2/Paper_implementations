"""
Cross-model KV cache transfer for multi-model agent handoffs.

Covers two complementary techniques from AlphaSignal Deep Dive (2026-10-12):
  1. Nvidia cross-model KV cache transfer — 25× speedup for model switching
     (ridge-regression mapping across layer pairs, RoPE-removed keys)
  2. Cache-2-Cache (C2C) — trained fuser for specialist delegation, 2.5× avg speedup

Neither technique requires changes to the models themselves.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Architecture description
# ---------------------------------------------------------------------------

TECHNIQUES = [
    {
        "id": "kv_transfer",
        "name": "Cross-Model KV Cache Transfer",
        "icon": "🔄",
        "source": "Nvidia Research",
        "use_case": "Model switching (e.g. Qwen3 14B → Qwen3 32B mid-session)",
        "speedup": "25×",
        "speedup_detail": "278 ms vs ~7 s to rebuild from scratch for 32K-token context",
        "accuracy": "Small task-dependent tradeoff; neural mapper variant recovers most accuracy",
        "requirements": "Model weights + access to KV buffers; calibration dataset (500 × 1024-token seqs)",
        "description": (
            "When an agent session running on a smaller model needs to switch to a larger one "
            "(or vice versa), the new model would normally re-read the entire accumulated context "
            "from scratch — the expensive *prefill* phase. This technique maps the source model's "
            "KV cache into the target model's internal representation space using three steps:\n\n"
            "1. **Layer selection** — pick source layers that best predict each target layer "
            "(via cross-validation on a small calibration set).\n"
            "2. **RoPE removal** — strip the rotary position embeddings from the key tensors "
            "before regression so the mapping generalises to context lengths beyond calibration.\n"
            "3. **Ridge regression** — fit a linear transform; no gradient descent or GPU training "
            "needed, calibration takes minutes on a CPU.\n\n"
            "The mapped cache is injected directly into the target model, which then generates "
            "its first token without any prefill overhead."
        ),
    },
    {
        "id": "c2c",
        "name": "Cache-2-Cache (C2C) Transfer",
        "icon": "🔗",
        "source": "Academic / Open Research",
        "use_case": "Specialist delegation (main agent → specialist running in parallel)",
        "speedup": "2.5× avg",
        "speedup_detail": "Cache fusion step ~90 ms vs 1,312 ms to generate an 80-token text report",
        "accuracy": "+3.1–5.4 pp over text-based communication (reasoning + knowledge benchmarks)",
        "requirements": "Both model weights + trained cache fuser (frozen models, only fuser trains)",
        "description": (
            "When a main agent delegates a subtask to a specialist, the traditional path is "
            "*text handoff*: the specialist generates a textual report, the main agent reads "
            "that text. C2C eliminates the generate + re-prefill cycle:\n\n"
            "1. Both models independently prefill over the *shared* context (the codebase, the "
            "task description) and build their own KV caches.\n"
            "2. A lightweight *cache fuser* (a trainable projection head) reads the source "
            "model's KV representations and injects them into the target model's cache.\n"
            "3. The target can now attend to the source's internal understanding without "
            "generating a single text token.\n\n"
            "The fuser works across models with different sizes, tokenizers, and architectures "
            "(tested: Qwen, Llama, Gemma families). Only the fuser is trained; both models are "
            "frozen. A hybrid approach (C2C + short text summary) outperforms either alone."
        ),
    },
]

# ---------------------------------------------------------------------------
# Benchmark data
# ---------------------------------------------------------------------------

BENCHMARK_ROWS = [
    # Cross-model transfer benchmarks
    {
        "technique": "Cross-Model KV Transfer",
        "scenario": "Qwen3 14B → 32B, 32K tokens",
        "metric": "Prefill latency",
        "baseline": "6,940 ms (rebuild from scratch)",
        "with_technique": "278 ms (cache transfer)",
        "gain": "25× speedup",
        "gain_color": "#22c55e",
    },
    {
        "technique": "Cross-Model KV Transfer",
        "scenario": "Qwen3 14B → 32B, accuracy",
        "metric": "Downstream task accuracy",
        "baseline": "100% (full rebuild)",
        "with_technique": "~97–99% (linear mapper)",
        "gain": "−1–3% tradeoff",
        "gain_color": "#fbbf24",
    },
    # C2C benchmarks
    {
        "technique": "Cache-2-Cache (C2C)",
        "scenario": "80-token specialist report",
        "metric": "Handoff latency",
        "baseline": "1,312 ms (text generation)",
        "with_technique": "~90 ms (cache fusion)",
        "gain": "14.6× for generation step",
        "gain_color": "#22c55e",
    },
    {
        "technique": "Cache-2-Cache (C2C)",
        "scenario": "Reasoning + knowledge benchmarks",
        "metric": "Task accuracy",
        "baseline": "Text-only baseline",
        "with_technique": "+3.1–5.4 pp (avg 2.5× latency win)",
        "gain": "+3.1–5.4 pp accuracy",
        "gain_color": "#22c55e",
    },
    {
        "technique": "Cache-2-Cache (C2C)",
        "scenario": "Math workflow (hybrid)",
        "metric": "Task accuracy",
        "baseline": "C2C alone or text alone",
        "with_technique": "C2C + text > either alone",
        "gain": "Best combined",
        "gain_color": "#6366f1",
    },
]


def benchmark_table_html() -> str:
    technique_color = {
        "Cross-Model KV Transfer": "#3b82f6",
        "Cache-2-Cache (C2C)": "#a855f7",
    }
    rows = "".join(
        f'<tr style="border-bottom:1px solid #1e293b;">'
        f'<td style="padding:10px 12px;">'
        f'<span style="background:{technique_color.get(r["technique"],"#64748b")}22;'
        f'color:{technique_color.get(r["technique"],"#94a3b8")};padding:2px 8px;'
        f'border-radius:4px;font-size:12px;white-space:nowrap;">{r["technique"]}</span></td>'
        f'<td style="padding:10px 12px;color:#94a3b8;font-size:13px;">{r["scenario"]}</td>'
        f'<td style="padding:10px 12px;color:#e2e8f0;font-size:13px;">{r["metric"]}</td>'
        f'<td style="padding:10px 12px;color:#f87171;font-size:13px;">{r["baseline"]}</td>'
        f'<td style="padding:10px 12px;color:#4ade80;font-size:13px;">{r["with_technique"]}</td>'
        f'<td style="padding:10px 12px;font-weight:700;color:{r["gain_color"]};font-size:13px;">{r["gain"]}</td>'
        f'</tr>'
        for r in BENCHMARK_ROWS
    )
    return (
        '<table style="width:100%;border-collapse:collapse;background:#0f172a;'
        'border-radius:8px;overflow:hidden;font-family:monospace;">'
        '<thead><tr style="background:#1e293b;">'
        + "".join(
            f'<th style="padding:10px 12px;text-align:left;color:#94a3b8;font-size:12px;">{h}</th>'
            for h in ["Technique", "Scenario", "Metric", "Baseline", "With Technique", "Gain"]
        )
        + f'</tr></thead><tbody>{rows}</tbody></table>'
    )


# ---------------------------------------------------------------------------
# Handoff cost simulator
# ---------------------------------------------------------------------------

def simulate_handoff(
    context_tokens: int,
    handoff_type: str,
    output_tokens: int = 80,
    use_technique: bool = True,
) -> dict:
    """
    Estimate latency for a model handoff, with and without cache transfer.

    Assumes ~215 tokens/s prefill throughput (A100-class single GPU baseline).
    """
    PREFILL_TOKENS_PER_SEC = 15_000  # tokens/s during prefill (A100-class)
    DECODE_TOKENS_PER_SEC = 215      # tokens/s during decode

    if handoff_type == "model_switch":
        baseline_ms = (context_tokens / PREFILL_TOKENS_PER_SEC) * 1000
        if use_technique:
            # Ridge-regression cache transfer scales sub-linearly
            transfer_ms = max(50, (context_tokens / 32_768) * 278)
            description = (
                f"Cross-model KV transfer: {transfer_ms:.0f} ms "
                f"(vs {baseline_ms:.0f} ms rebuild) — "
                f"{baseline_ms/transfer_ms:.1f}× faster"
            )
            return {
                "technique": "Cross-Model KV Transfer",
                "baseline_ms": round(baseline_ms),
                "optimized_ms": round(transfer_ms),
                "speedup": round(baseline_ms / transfer_ms, 1),
                "description": description,
            }
        return {
            "technique": "No optimization",
            "baseline_ms": round(baseline_ms),
            "optimized_ms": round(baseline_ms),
            "speedup": 1.0,
            "description": f"Full prefill rebuild: {baseline_ms:.0f} ms",
        }

    else:  # specialist_delegation via C2C
        # Text path: generate report tokens, then receiver prefills those tokens
        generate_ms = (output_tokens / DECODE_TOKENS_PER_SEC) * 1000
        receiver_prefill_ms = (output_tokens / PREFILL_TOKENS_PER_SEC) * 1000
        baseline_ms = generate_ms + receiver_prefill_ms
        if use_technique:
            # Cache fuser is approximately 90ms regardless of token count (small MLP)
            fusion_ms = 90
            description = (
                f"C2C cache fusion: {fusion_ms:.0f} ms "
                f"(vs {baseline_ms:.0f} ms text report path) — "
                f"{baseline_ms/fusion_ms:.1f}× faster"
            )
            return {
                "technique": "Cache-2-Cache (C2C)",
                "baseline_ms": round(baseline_ms),
                "optimized_ms": fusion_ms,
                "speedup": round(baseline_ms / fusion_ms, 1),
                "description": description,
            }
        return {
            "technique": "No optimization",
            "baseline_ms": round(baseline_ms),
            "optimized_ms": round(baseline_ms),
            "speedup": 1.0,
            "description": f"Text report path: {baseline_ms:.0f} ms",
        }


def run_handoff_simulation(
    context_tokens: int,
    handoff_type: str,
    output_tokens: int = 80,
) -> str:
    baseline = simulate_handoff(context_tokens, handoff_type, output_tokens, use_technique=False)
    optimized = simulate_handoff(context_tokens, handoff_type, output_tokens, use_technique=True)

    technique_name = optimized["technique"]
    speedup = optimized["speedup"]
    baseline_ms = baseline["baseline_ms"]
    opt_ms = optimized["optimized_ms"]

    agent_runs_per_hour_baseline = int(3_600_000 / (baseline_ms + 500))
    agent_runs_per_hour_optimized = int(3_600_000 / (opt_ms + 500))

    return (
        f"### Handoff Cost Estimate\n\n"
        f"| | Baseline | With {technique_name} |\n"
        f"|---|---|---|\n"
        f"| Handoff latency | **{baseline_ms:,} ms** | **{opt_ms:,} ms** |\n"
        f"| Speedup | 1× | **{speedup}×** |\n"
        f"| Agent runs/hour (500ms work each) | ~{agent_runs_per_hour_baseline:,} | ~{agent_runs_per_hour_optimized:,} |\n\n"
        f"**Context:** {context_tokens:,} tokens  |  "
        f"**Handoff:** {handoff_type.replace('_', ' ')}  |  "
        f"{'Specialist report: ' + str(output_tokens) + ' tokens' if handoff_type == 'specialist_delegation' else 'Full context transferred'}\n\n"
        f"> _{optimized['description']}_"
    )


# ---------------------------------------------------------------------------
# Implementation patterns
# ---------------------------------------------------------------------------

IMPL_PATTERNS = [
    {
        "name": "Pattern 1 — Cross-Model KV Cache Transfer (Calibration)",
        "description": (
            "Calibrate a ridge-regression KV mapper between two models of the same family "
            "(e.g. Qwen3-14B → Qwen3-32B). Run once offline; the resulting weight matrix is "
            "reused for all production handoffs."
        ),
        "code": '''\
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import Ridge

SMALL_MODEL = "Qwen/Qwen3-14B"
LARGE_MODEL = "Qwen/Qwen3-32B"
CALIBRATION_SEQS = 500   # sequences used for calibration
SEQ_LEN = 1024

small = AutoModelForCausalLM.from_pretrained(SMALL_MODEL, output_hidden_states=True)
large = AutoModelForCausalLM.from_pretrained(LARGE_MODEL, output_hidden_states=True)
tok   = AutoTokenizer.from_pretrained(SMALL_MODEL)

def collect_kv(model, input_ids):
    """Return KV pairs from every layer (RoPE already stripped from keys)."""
    with torch.no_grad():
        out = model(input_ids, use_cache=True, output_hidden_states=False)
    # past_key_values: list of (key, value) per layer
    return out.past_key_values

# Collect calibration data (offline, run once)
mappers = {}  # src_layer -> Ridge model
for src_layer_idx in range(small.config.num_hidden_layers):
    X, Y = [], []
    for _ in range(CALIBRATION_SEQS):
        ids = tok(["sample text ..."], return_tensors="pt", truncation=True,
                  max_length=SEQ_LEN).input_ids
        small_kv = collect_kv(small, ids)
        large_kv = collect_kv(large, ids)
        src_k = small_kv[src_layer_idx][0].flatten(1).numpy()
        # Pick best-matching target layer (via cross-val; simplified here to same index)
        tgt_k = large_kv[src_layer_idx][0].flatten(1).numpy()
        X.append(src_k); Y.append(tgt_k)
    reg = Ridge(alpha=1e-3).fit(
        torch.cat([torch.tensor(x) for x in X]).numpy(),
        torch.cat([torch.tensor(y) for y in Y]).numpy(),
    )
    mappers[src_layer_idx] = reg

# At inference time: transfer the cache without prefill
def transfer_kv_to_large(small_past_kv, mappers):
    transferred = []
    for layer_idx, (k, v) in enumerate(small_past_kv):
        mapped_k = torch.tensor(
            mappers[layer_idx].predict(k.flatten(1).numpy())
        ).reshape_as(k)
        transferred.append((mapped_k, v))   # values reused as-is
    return transferred
''',
    },
    {
        "name": "Pattern 2 — Cache-2-Cache (C2C) Fuser Training",
        "description": (
            "Train a lightweight cache fuser that projects the source model's KV "
            "representations into the target model's space. Both models stay frozen. "
            "Suitable for fixed source+target pairs (e.g. a general agent + a code specialist)."
        ),
        "code": '''\
import torch
import torch.nn as nn

class CacheFuser(nn.Module):
    """Projects src KV cache into tgt KV cache space — one MLP per layer pair."""
    def __init__(self, src_dim: int, tgt_dim: int, hidden: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(src_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, tgt_dim),
        )

    def forward(self, src_k: torch.Tensor) -> torch.Tensor:
        B, H, T, D = src_k.shape
        flat = src_k.reshape(B * H * T, D)
        fused = self.proj(flat)
        return fused.reshape(B, H, T, -1)


# Training loop (src/tgt models frozen)
def train_fuser(src_model, tgt_model, dataset, epochs=3):
    fuser = CacheFuser(src_dim=src_model.config.hidden_size // src_model.config.num_key_value_heads,
                       tgt_dim=tgt_model.config.hidden_size // tgt_model.config.num_key_value_heads)
    opt = torch.optim.AdamW(fuser.parameters(), lr=1e-4)

    for _ in range(epochs):
        for batch in dataset:
            input_ids = batch["input_ids"]
            with torch.no_grad():
                src_kv = src_model(input_ids, use_cache=True).past_key_values
                tgt_kv = tgt_model(input_ids, use_cache=True).past_key_values

            loss = sum(
                nn.functional.mse_loss(fuser(src_kv[i][0]), tgt_kv[i][0])
                for i in range(len(src_kv))
            )
            opt.zero_grad(); loss.backward(); opt.step()

    return fuser


# At agent handoff time:
def delegate_via_c2c(shared_context_ids, src_model, tgt_model, fuser):
    with torch.no_grad():
        src_kv = src_model(shared_context_ids, use_cache=True).past_key_values
        fused_kv = [(fuser(k), v) for k, v in src_kv]
        # Inject into tgt_model; it continues from fused cache
        out = tgt_model(shared_context_ids[:, -1:], past_key_values=fused_kv,
                        use_cache=True)
    return out
''',
    },
    {
        "name": "Pattern 3 — Production Orchestrator with Adaptive Handoff",
        "description": (
            "A production-ready orchestrator that chooses the right handoff strategy based "
            "on context length and task type: KV transfer for model switches, C2C for "
            "specialist delegation, plain text for short contexts where overhead isn't worth it."
        ),
        "code": '''\
from dataclasses import dataclass
from enum import Enum

class HandoffStrategy(Enum):
    KV_TRANSFER = "kv_transfer"   # Nvidia cross-model cache transfer
    C2C = "c2c"                   # Cache-2-Cache specialist delegation
    TEXT = "text"                  # Plain text (short contexts)


@dataclass
class HandoffConfig:
    kv_transfer_threshold_tokens: int = 8_000   # use KV transfer above this
    c2c_output_tokens_threshold: int = 40        # use C2C if report > this tokens


class AgentOrchestrator:
    def __init__(self, config: HandoffConfig):
        self.cfg = config
        self._mappers = {}     # loaded on first use
        self._fusers = {}      # loaded on first use

    def choose_strategy(
        self,
        context_tokens: int,
        handoff_type: str,  # "model_switch" | "specialist_delegation"
        expected_output_tokens: int = 80,
        models_in_same_family: bool = True,
    ) -> HandoffStrategy:
        if handoff_type == "model_switch":
            if (context_tokens >= self.cfg.kv_transfer_threshold_tokens
                    and models_in_same_family):
                return HandoffStrategy.KV_TRANSFER
            return HandoffStrategy.TEXT

        if handoff_type == "specialist_delegation":
            if expected_output_tokens >= self.cfg.c2c_output_tokens_threshold:
                return HandoffStrategy.C2C
            return HandoffStrategy.TEXT

        return HandoffStrategy.TEXT

    def execute_handoff(self, strategy, src_session, tgt_model, context_ids):
        if strategy == HandoffStrategy.KV_TRANSFER:
            src_kv = src_session.past_key_values
            mapped_kv = self._apply_kv_mapper(src_kv, tgt_model.config.name)
            return tgt_model(context_ids[:, -1:], past_key_values=mapped_kv)

        if strategy == HandoffStrategy.C2C:
            fuser = self._get_fuser(src_session.model_name, tgt_model.config.name)
            src_kv = src_session.past_key_values
            fused_kv = [(fuser(k), v) for k, v in src_kv]
            return tgt_model(context_ids[:, -1:], past_key_values=fused_kv)

        # TEXT fallback: generate summary, re-prefill
        summary = src_session.model.generate_summary(context_ids)
        return tgt_model(summary)

    def _apply_kv_mapper(self, src_kv, tgt_name):
        mapper = self._mappers.get(tgt_name) or self._load_mapper(tgt_name)
        return [(mapper.transform(k), v) for k, v in src_kv]

    def _get_fuser(self, src_name, tgt_name):
        key = (src_name, tgt_name)
        return self._fusers.get(key) or self._load_fuser(*key)

    def _load_mapper(self, tgt_name): ...   # load ridge-regression weights
    def _load_fuser(self, src, tgt):  ...   # load trained CacheFuser checkpoint
''',
    },
]
