"""
DeepSeek V4.1-Flash KV Cache Efficiency — POC (Tab 35)
Source: AlphaSignal deep-dive, 2026-09-13 · news@alphasignal.ai
Score: 9/10

Techniques:
  CED — Causal Encoder-Decoder (split prefill / generate layers)
  SWA + Bounded Replay — sliding window attention, cheap KV reconstruction
  CSA2 — Compressed Sparse Attention 2 (Full / Reindex / Reuse layer modes)
  Hierarchical Sparse Indexer — bounded search cost over 1M-token contexts
  FP4 — 4-bit KV compression
"""

from __future__ import annotations

# ─── KV evolution data ───────────────────────────────────────────────────────

KV_EVOLUTION: list[dict] = [
    {"model": "DeepSeek V3.2",       "bytes_per_token": 48_000, "label": "48,000 B (~48 KB)", "color": "#ef4444"},
    {"model": "DeepSeek V4-Flash",   "bytes_per_token":  3_514, "label": "3,514 B",            "color": "#f97316"},
    {"model": "DeepSeek V4.1-Flash", "bytes_per_token":    890, "label": "890 B",               "color": "#22c55e"},
]

# ─── CSA2 mode definitions ────────────────────────────────────────────────────

CSA2_MODES: list[dict] = [
    {
        "name": "Full",
        "icon": "🔵",
        "kv_cost": "High — creates fresh KV pairs for entire context",
        "search_cost": "High — scans full context, creates 2048-block candidate pool",
        "description": (
            "Creates a fresh KV state. Runs expensive full-context attention and "
            "indexes all tokens into a candidate pool (≤2,048 blocks). "
            "The first decoder layer is always Full."
        ),
    },
    {
        "name": "Reindex",
        "icon": "🟡",
        "kv_cost": "Zero — borrows KV from nearest Full layer above",
        "search_cost": "Medium — searches only within the 2,048-block pool",
        "description": (
            "Shares the KV state from the nearest Full layer. Runs its own sparse "
            "search over the pool to select a different subset (≤512 positions). "
            "No new KV buffer allocated."
        ),
    },
    {
        "name": "Reuse",
        "icon": "🟢",
        "kv_cost": "Zero — shares KV and selected positions",
        "search_cost": "Zero — reuses index from a prior layer",
        "description": (
            "Shares both the KV state and the selected attention positions from a "
            "prior layer. Near-zero marginal compute cost. "
            "Allows many cheap decoder passes."
        ),
    },
]

# ─── Cost comparison ─────────────────────────────────────────────────────────

COST_MODELS: dict[str, dict] = {
    "GPT-5.6 (frontier)":    {"input_per_m": 5.00, "output_per_m": 15.00, "ai_index": 42},
    "Gemini 3.8 Flash High":  {"input_per_m": 1.00, "output_per_m":  4.00, "ai_index": 41},
    "DeepSeek V4.1-Flash":    {"input_per_m": 0.30, "output_per_m":  1.20, "ai_index": 40},
    "DeepSeek V4-Flash":      {"input_per_m": 0.80, "output_per_m":  2.40, "ai_index": 37},
}

# ─── Rendering helpers ────────────────────────────────────────────────────────

def kv_evolution_html() -> str:
    max_v = KV_EVOLUTION[0]["bytes_per_token"]
    rows = "".join(
        f'<tr>'
        f'<td style="padding:8px 16px 8px 0;white-space:nowrap;color:#cbd5e1;">{e["model"]}</td>'
        f'<td style="padding:8px 0;">'
        f'<div style="background:#1e293b;border-radius:4px;height:26px;width:400px;overflow:hidden;">'
        f'<div style="background:{e["color"]};width:{e["bytes_per_token"]/max_v*100:.1f}%;height:100%;'
        f'border-radius:4px;display:flex;align-items:center;padding-left:8px;'
        f'color:#fff;font-size:12px;font-weight:600;">{e["label"]}</div></div></td>'
        f'</tr>'
        for e in KV_EVOLUTION
    )
    return (
        '<table style="border-collapse:collapse;">'
        f'<thead><tr><th style="text-align:left;padding-bottom:8px;color:#94a3b8;">Model</th>'
        f'<th style="text-align:left;padding-bottom:8px;color:#94a3b8;">KV bytes / token (global)</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    )


def kv_memory_html(context_len: int) -> str:
    rows = "".join(
        f'<tr style="border-bottom:1px solid #1e293b;">'
        f'<td style="padding:10px 14px;color:#e2e8f0;">{e["model"]}</td>'
        f'<td style="padding:10px 14px;color:{e["color"]};font-weight:700;">'
        f'{e["bytes_per_token"] * context_len / 1_000_000:.1f} MB</td>'
        f'</tr>'
        for e in KV_EVOLUTION
    )
    return (
        f'<p style="color:#94a3b8;font-size:13px;">Global KV memory for '
        f'<strong>{context_len:,}</strong> tokens (one sequence):</p>'
        '<table style="width:100%;border-collapse:collapse;background:#0f172a;border-radius:8px;overflow:hidden;">'
        '<thead><tr style="background:#1e293b;">'
        '<th style="padding:10px 14px;text-align:left;color:#94a3b8;">Model</th>'
        '<th style="padding:10px 14px;text-align:left;color:#94a3b8;">Memory</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )


def csa2_table_html() -> str:
    rows = "".join(
        f'<tr style="border-bottom:1px solid #1e293b;">'
        f'<td style="padding:12px 8px;font-size:20px;text-align:center;">{m["icon"]}</td>'
        f'<td style="padding:12px 14px;font-weight:700;color:#e2e8f0;">{m["name"]}</td>'
        f'<td style="padding:12px 14px;color:#94a3b8;font-size:13px;">{m["description"]}</td>'
        f'<td style="padding:12px 14px;color:#fbbf24;font-size:12px;">{m["kv_cost"]}</td>'
        f'<td style="padding:12px 14px;color:#34d399;font-size:12px;">{m["search_cost"]}</td>'
        f'</tr>'
        for m in CSA2_MODES
    )
    return (
        '<table style="width:100%;border-collapse:collapse;background:#0f172a;border-radius:8px;overflow:hidden;">'
        '<thead><tr style="background:#1e293b;">'
        '<th style="padding:10px;text-align:center;color:#94a3b8;"></th>'
        '<th style="padding:10px 14px;text-align:left;color:#94a3b8;">Mode</th>'
        '<th style="padding:10px 14px;text-align:left;color:#94a3b8;">How it works</th>'
        '<th style="padding:10px 14px;text-align:left;color:#94a3b8;">KV cost</th>'
        '<th style="padding:10px 14px;text-align:left;color:#94a3b8;">Search cost</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )


def cost_table_html(tokens_m: float) -> str:
    output_ratio = 0.20
    rows = "".join(
        f'<tr style="border-bottom:1px solid #1e293b;">'
        f'<td style="padding:10px 14px;color:#e2e8f0;">{name}</td>'
        f'<td style="padding:10px 14px;color:#94a3b8;">{d["ai_index"]}</td>'
        f'<td style="padding:10px 14px;color:#60a5fa;">${d["input_per_m"]:.2f}</td>'
        f'<td style="padding:10px 14px;color:#c084fc;">${d["output_per_m"]:.2f}</td>'
        f'<td style="padding:10px 14px;font-weight:700;color:#fbbf24;">'
        f'${d["input_per_m"]*tokens_m + d["output_per_m"]*tokens_m*output_ratio:.2f}</td>'
        f'</tr>'
        for name, d in COST_MODELS.items()
    )
    return (
        f'<p style="color:#94a3b8;font-size:13px;">Cost for <strong>{tokens_m:.0f}M</strong> input tokens '
        f'(+{tokens_m*output_ratio:.0f}M output tokens at 20% output ratio):</p>'
        '<table style="width:100%;border-collapse:collapse;background:#0f172a;border-radius:8px;overflow:hidden;">'
        '<thead><tr style="background:#1e293b;">'
        '<th style="padding:10px 14px;text-align:left;color:#94a3b8;">Model</th>'
        '<th style="padding:10px 14px;text-align:left;color:#94a3b8;">AI Index</th>'
        '<th style="padding:10px 14px;text-align:left;color:#94a3b8;">$/M input</th>'
        '<th style="padding:10px 14px;text-align:left;color:#94a3b8;">$/M output</th>'
        '<th style="padding:10px 14px;text-align:left;color:#94a3b8;">Total $</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )
