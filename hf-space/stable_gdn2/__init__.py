"""
Stable Gated DeltaNet-2: recurrent state instability (Tab 59)
Source: Hugging Face Daily Papers · 2026-11-27 · arXiv 2611.34051
Score: 8/10

The paper traces long-context training failures of Gated DeltaNet-2 to non-normal erase operators
(read and modify directions differ), whose directional gains accumulate across tokens and make the
recurrent state grow exponentially even without writes. Symmetric erasure keeps propagation
non-expansive; decay-regulated residual writing keeps writing flexible.

This demo is an ILLUSTRATION with random toy matrices (numpy). It is not the paper's model, kernels or
data; it only shows how non-normal erase operators amplify a state while symmetric ones cannot.
"""

from __future__ import annotations

import numpy as np

FINDINGS = [
    {"name": "Erasure in a delta rule is a rank-one update",
     "body": "Each token applies E = I - b * u v^T to the state. When the direction that is read (v) "
             "differs from the direction that is modified (u), E is non-normal and can have norm above 1."},
    {"name": "Gains accumulate across tokens",
     "body": "A norm slightly above 1 per token compounds over a long context, so the state can grow "
             "exponentially even if nothing new is written. This matches the training failures at long context."},
    {"name": "Symmetric erasure is non-expansive",
     "body": "With u = v (unit norm) and 0 <= b <= 1, E is an orthogonal projection blend with norm at most 1, "
             "so the state cannot be amplified by erasure alone. The paper reports an O(sqrt(T)) state bound."},
    {"name": "Illustration only",
     "body": "All numbers in this tab come from random toy matrices, not from the paper's models or benchmarks."},
]

IMPL_PATTERNS = [
    {"name": "Asymmetric erase (unstable)",
     "description": "Read direction k_r and modify direction k_w are different vectors.",
     "code": "def erase_asym(S, k_r, k_w, beta):\n    # S: (d, d); removes beta * (S k_r) along k_w\n    return S - beta * np.outer(S @ k_r, k_w)"},
    {"name": "Symmetric erase (non-expansive)",
     "description": "Use one unit vector for both reading and modifying; keep 0 <= beta <= 1.",
     "code": "def erase_sym(S, k, beta):\n    k = k / (np.linalg.norm(k) + 1e-8)\n    return S - beta * np.outer(S @ k, k)"},
]

MODES = ("Asymmetric erase", "Symmetric erase")


def _validate(dim, steps, beta, skew, write_scale, seed):
    try:
        d, t, b, c, w, s = int(dim), int(steps), float(beta), float(skew), float(write_scale), int(seed)
    except (TypeError, ValueError):
        return None
    if not (2 <= d <= 64 and 10 <= t <= 5000 and 0.0 < b <= 2.0 and 0.0 <= c <= 1.0 and 0.0 <= w <= 5.0):
        return None
    return d, t, b, c, w, s


def _unit(rng, d):
    v = rng.standard_normal(d)
    return v / (np.linalg.norm(v) + 1e-12)


def run(mode, dim=16, steps=500, beta=1.5, skew=0.8, write_scale=0.0, seed=0):
    """Return list of state Frobenius norms over time (len steps+1), or None for invalid input."""
    args = _validate(dim, steps, beta, skew, write_scale, seed)
    if args is None or mode not in MODES:
        return None
    d, t, b, c, w, s = args
    rng = np.random.default_rng(s)
    S = np.eye(d)
    norms = [float(np.linalg.norm(S))]
    for _ in range(t):
        k_r = _unit(rng, d)
        if mode == "Symmetric erase":
            k_w = k_r
            b_eff = min(b, 1.0)
        else:
            # modify direction rotated away from read direction by `skew`, beta unconstrained
            other = _unit(rng, d)
            k_w = k_r + c * other
            b_eff = b
        S = S - b_eff * np.outer(S @ k_r, k_w)
        if w > 0:
            v = rng.standard_normal(d) * w / np.sqrt(d)
            S = S + np.outer(v, _unit(rng, d))
        norms.append(float(np.linalg.norm(S)))
        if not np.isfinite(norms[-1]) or norms[-1] > 1e150:
            norms[-1] = 1e150
            break
    return norms


def compare(dim=16, steps=500, beta=1.5, skew=0.8, write_scale=0.0, seed=0):
    """Return (markdown, rows). rows: [step, asymmetric norm, symmetric norm, sqrt(T) reference]."""
    a = run("Asymmetric erase", dim, steps, beta, skew, write_scale, seed)
    sy = run("Symmetric erase", dim, steps, beta, skew, write_scale, seed)
    if a is None or sy is None:
        return ("Use dimension 2-64, 10-5000 steps, beta in (0, 2], skew 0-1 and write scale 0-5.", [])
    t = int(steps)
    marks = sorted({1, 10, 50, 100, 250, 500, 1000, 2500, 5000} & set(range(1, t + 1)) | {t})
    rows = []
    for m in marks:
        ai = a[m] if m < len(a) else a[-1]
        rows.append([m, float(f"{ai:.4g}"), float(f"{sy[m]:.4g}"), round(float(np.sqrt(m)), 2)])
    verdict = (f"After {t} tokens the asymmetric state norm is {a[-1]:.3g} against {sy[-1]:.3g} "
               f"for symmetric erasure (toy matrices, illustration only).")
    return verdict, rows


def gain_table(dim=16, beta=1.5, n=2000, seed=0):
    """Return (markdown, rows): per-token erase gain (spectral norm) as skew grows."""
    try:
        d, b, n_, s = int(dim), float(beta), int(n), int(seed)
    except (TypeError, ValueError):
        return ("Invalid parameters.", [])
    if not (2 <= d <= 64 and 0.0 < b <= 2.0 and 10 <= n_ <= 20000):
        return ("Use dimension 2-64, beta in (0, 2] and 10-20000 samples.", [])
    rng = np.random.default_rng(s)
    rows = []
    for c in (0.0, 0.25, 0.5, 1.0):
        gains = []
        for _ in range(min(n_, 500)):
            k_r = _unit(rng, d)
            k_w = k_r + c * _unit(rng, d)
            E = np.eye(d) - b * np.outer(k_w, k_r)
            gains.append(np.linalg.norm(E, 2))
        g = np.array(gains)
        rows.append([c, round(float(g.mean()), 3), round(float(g.max()), 3), round(100 * float((g > 1.0001).mean()), 1)])
    return ("Erase gain above 1 appears as soon as read and modify directions differ "
            "(toy matrices, illustration only).", rows)
