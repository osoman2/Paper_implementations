"""
Multi-Branch Transformer: parallel blocks with a Joint-KV cache (Tab 56)
Source: Hugging Face Daily Papers · 2026-11-18 · https://arxiv.org/abs/2611.19800
Score: 8/10

Only the abstract was used. The KV-cache arithmetic below is a back-of-envelope illustration of
which layers store a cache, not the paper's measurements (which report up to 34% less KV storage
and up to 51.4% higher decoding throughput on one H200).
"""

FINDINGS = [
    {"name": "Branching as a third scaling axis",
     "body": "Instead of only widening or deepening, the input goes through shared initial layers, "
             "then independently parameterized branches of sequential blocks in parallel, then a "
             "fusion step and shared final layers."},
    {"name": "Three variants",
     "body": "Wide-Shallow keeps block width, Narrow-Deep keeps sequential depth, Balanced "
             "interpolates. 2-branch Wide-Shallow beats sequential Transformers at matched "
             "parameters and compute; Balanced improves further."},
    {"name": "Joint-KV",
     "body": "A shared KV cache is built from the branch representations at each depth, so the "
             "cache does not grow with the number of branches. The paper reports up to 34% less "
             "full-model KV storage and up to 51.4% higher decoding throughput (H200)."},
]

VARIANT_ROWS = [
    ["Sequential baseline", "One chain of L blocks", "L KV layers"],
    ["Multi-Branch (per-branch KV)", "Shared head, B parallel chains, shared tail", "head + B x branch + tail"],
    ["Multi-Branch + Joint-KV", "Same, one shared cache per depth", "head + branch + tail"],
]


def _num(v, lo, hi, default):
    try:
        return int(min(max(int(v), lo), hi))
    except (TypeError, ValueError):
        return default


def kv_layers(head=4, branch_depth=6, branches=2, tail=2):
    """Number of KV-cache layers stored per token for each design."""
    head = _num(head, 0, 64, 4)
    branch_depth = _num(branch_depth, 1, 64, 6)
    branches = _num(branches, 1, 8, 2)
    tail = _num(tail, 0, 64, 2)
    seq_equiv = head + branch_depth * branches + tail  # same block count in one chain
    per_branch = head + branch_depth * branches + tail
    joint = head + branch_depth + tail
    return head, branch_depth, branches, tail, seq_equiv, per_branch, joint


def kv_calc(head=4, branch_depth=6, branches=2, tail=2, context=128000, kv_bytes_per_layer_token=2048):
    """Illustrative KV memory for a given context length. Not the paper's measurements."""
    try:
        context = int(min(max(int(context), 1), 10_000_000))
        per = float(kv_bytes_per_layer_token)
        if per <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return "Enter numeric values.", []
    h, d, b, t, seq, pb, joint = kv_layers(head, branch_depth, branches, tail)
    gb = lambda layers: layers * context * per / 1e9
    saving = 1 - joint / pb
    rows = [
        ["Sequential chain, same block count", seq, f"{gb(seq):.2f}"],
        ["Multi-Branch, per-branch KV", pb, f"{gb(pb):.2f}"],
        ["Multi-Branch + Joint-KV", joint, f"{gb(joint):.2f}"],
    ]
    msg = (f"Illustration only (layer counting, not the paper's numbers). Joint-KV stores "
           f"{saving:.0%} fewer KV layers than per-branch caching for this shape.")
    return msg, rows


def advice(branches=2, context=128000):
    try:
        b, c = int(branches), int(context)
    except (TypeError, ValueError):
        return "Enter numeric values."
    if b <= 1:
        return "One branch is a plain sequential Transformer: Joint-KV gives no saving."
    if c < 8000:
        return "At short contexts KV storage is small; branching mainly helps quality per compute."
    return "Long context with several branches: sharing the KV cache (Joint-KV) matters most here."


IMPL_PATTERNS = [
    {"name": "Branch block (PyTorch sketch)",
     "description": "Parallel branches over a shared head, fused by a mean before the shared tail.",
     "code": ("import torch, torch.nn as nn\n\n"
              "class Branches(nn.Module):\n"
              "    def __init__(self, make_block, depth, n_branches):\n"
              "        super().__init__()\n"
              "        self.branches = nn.ModuleList(\n"
              "            nn.Sequential(*[make_block() for _ in range(depth)])\n"
              "            for _ in range(n_branches))\n\n"
              "    def forward(self, x):\n"
              "        return torch.stack([b(x) for b in self.branches]).mean(0)\n")},
]
