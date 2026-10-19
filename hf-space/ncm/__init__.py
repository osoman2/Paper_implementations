"""
Next-Concept Modeling (NCM) — POC (Tab 52)
Source: AlphaSignal Sunday Deep Dive · 2026-10-18 · news@alphasignal.ai
Score: 8/10

NCM (Stanford) replaces token-by-token generation with coarse-to-fine generation of a
hierarchy of discrete codes: a multiscale autoencoder learns the codes, a transformer
predicts all codes of one level in parallel, so decoding needs one step per level
instead of one step per token. This module holds a toy, dependency-free illustration
of the step/latency arithmetic, the coarse-to-fine refinement, and MBR selection.
"""

import re

CARDS = [
    {
        "name": "Two-stage training",
        "body": (
            "**Stage 1 - represent:** a multiscale autoencoder compresses text into a hierarchy of "
            "discrete codes (few codes at coarse levels, more at finer levels) and reconstructs the "
            "sequence from them.\n\n"
            "**Stage 2 - generate:** a transformer predicts the codes of each level conditioned on the "
            "reconstruction from earlier levels. With teacher forcing, all levels train in one forward pass."
        ),
    },
    {
        "name": "Decoding cost",
        "body": (
            "Autoregressive: one sequential step per output token. Diffusion LMs: many denoising steps, "
            "each over the full sequence. NCM: **one step per hierarchy level**, each predicting that "
            "level's codes in parallel, and early levels are short sequences."
        ),
    },
    {
        "name": "Reported results (paper-scale, ~100M generators)",
        "body": (
            "- 10-31x faster than strong diffusion LM baselines at comparable or better quality "
            "(21x on LM1B, H100)\n- 4-5x fewer training tokens in selected benchmarks\n"
            "- Better than the strongest baseline on hard Sudoku with fewer model evaluations\n"
            "- ~7% higher ROUGE-L on summarization when spending the saved time on MBR decoding\n"
            "- Scaled to 725M parameters; MIT code and checkpoints released\n\n"
            "Caveat: early-stage, small models. No evidence yet at LLM scale."
        ),
    },
]

IMPL_PATTERNS = [
    {
        "name": "Spend saved latency on MBR decoding",
        "description": "Fast samplers can afford N candidates. Pick the one most similar to the others.",
        "code": '''def mbr_select(candidates, sim):
    scores = [
        sum(sim(c, o) for j, o in enumerate(candidates) if j != i) / (len(candidates) - 1)
        for i, c in enumerate(candidates)
    ]
    return candidates[max(range(len(candidates)), key=scores.__getitem__)]
''',
    },
    {
        "name": "Coarse-to-fine decode loop (sketch)",
        "description": "One sequential step per level; all codes inside a level are sampled in parallel.",
        "code": '''codes = []
for level in range(num_levels):
    logits = generator(context=codes)          # predicts every code of this level at once
    codes.append(logits.argmax(-1))
    draft = decoder(codes)                      # usable (rough) text after every level
text = decoder(codes)
''',
    },
]


def decode_steps(n_tokens, levels, diffusion_steps):
    """Sequential steps needed by each decoding paradigm."""
    return {"Autoregressive": n_tokens, "Diffusion LM": diffusion_steps, "NCM": levels}


def latency_table(n_tokens, levels, diffusion_steps, step_ms=20.0):
    """Markdown table of steps and relative token-positions processed (toy cost model).

    AR processes 1 new position per step (KV cache), diffusion the full sequence per step,
    NCM a level of length n_tokens / 2^(levels-l) per step (geometric growth to full length).
    """
    n_tokens, levels, diffusion_steps = int(n_tokens), int(levels), int(diffusion_steps)
    ar_work = n_tokens
    dlm_work = diffusion_steps * n_tokens
    ncm_work = sum(max(1, round(n_tokens / 2 ** (levels - 1 - l))) for l in range(levels))
    steps = decode_steps(n_tokens, levels, diffusion_steps)
    work = {"Autoregressive": ar_work, "Diffusion LM": dlm_work, "NCM": ncm_work}
    rows = ["| Method | Sequential steps | Positions computed | Est. latency (ms) |", "|---|---|---|---|"]
    for k in steps:
        rows.append(f"| {k} | {steps[k]} | {work[k]} | {steps[k] * step_ms:.0f} |")
    speed = steps["Diffusion LM"] / max(1, steps["NCM"])
    return "\n".join(rows) + (
        f"\n\nNCM needs **{speed:.1f}x fewer sequential steps** than this diffusion setting. "
        "Toy model: assumes equal per-step latency; real speedups depend on hardware and model size."
    )


def coarse_to_fine(text, levels):
    """Toy illustration: level k reveals every 2^(L-k)-th word; later levels fill the gaps.

    This is NOT the learned autoencoder, only a picture of progressive refinement.
    """
    words = text.split()
    levels = max(1, int(levels))
    out = []
    for l in range(levels):
        stride = 2 ** (levels - 1 - l)
        shown = [w if i % stride == 0 else "___" for i, w in enumerate(words)]
        out.append(f"**Level {l + 1}** ({sum(1 for w in shown if w != '___')} of {len(words)} words): "
                   + " ".join(shown))
    return "\n\n".join(out)


def _tokens(s):
    return re.findall(r"\w+", s.lower())


def token_f1(a, b):
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    common = 0
    pool = list(tb)
    for t in ta:
        if t in pool:
            pool.remove(t)
            common += 1
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def mbr_select(candidates_text):
    """Minimum Bayes risk: choose the candidate with the highest mean F1 to all others."""
    cands = [c.strip() for c in (candidates_text or "").splitlines() if c.strip()]
    if len(cands) < 2:
        return "Enter at least 2 candidates, one per line."
    scores = [sum(token_f1(c, o) for j, o in enumerate(cands) if j != i) / (len(cands) - 1)
              for i, c in enumerate(cands)]
    best = max(range(len(cands)), key=scores.__getitem__)
    rows = ["| # | Candidate | Mean F1 vs others |", "|---|---|---|"]
    for i, (c, s) in enumerate(zip(cands, scores)):
        mark = " **(selected)**" if i == best else ""
        rows.append(f"| {i + 1} | {c}{mark} | {s:.3f} |")
    return "\n".join(rows)
