"""
The Resolution Hypothesis: can an agent-evolution loop tell a better candidate from a worse one? (Tab 57)
Source: Hugging Face Daily Papers · 2026-11-19 · arXiv 2611.23068
Score: 8/10

The paper argues that agent evolution is bottlenecked by noisy evaluations that cannot reliably
identify better candidates, so progress needs improvements to be detectable and accumulable.

This demo is an ILLUSTRATION on a simulated evaluator (Bernoulli task outcomes). It is not the paper's
method or data; it shows the statistical idea: the smallest improvement an evaluator can resolve
shrinks only with the square root of the number of evaluation tasks.
"""

from __future__ import annotations

import math

import numpy as np

FINDINGS = [
    {"name": "Evaluation noise sets a resolution limit",
     "body": "If a candidate is scored on n tasks, its measured pass rate has a standard error of about "
             "sqrt(p(1-p)/n). Improvements below that scale cannot be told from luck."},
    {"name": "Undetected gains cannot accumulate",
     "body": "An evolution loop that keeps the candidate with the best noisy score will often keep a "
             "worse one, so true gains are lost or reversed over generations."},
    {"name": "Illustration only",
     "body": "All numbers in this tab come from a simulated evaluator, not from the paper."},
]


def min_detectable(n_tasks, base_rate=0.5, z=1.645 + 0.84):
    """Smallest true gain (pass-rate points) a paired-free two-sample test resolves with ~80% power."""
    try:
        n = int(n_tasks)
        p = float(base_rate)
    except (TypeError, ValueError):
        return None
    if n < 1 or not 0.01 <= p <= 0.99:
        return None
    return z * math.sqrt(2 * p * (1 - p) / n)


def _validate(base_rate, gain, n_tasks, n_trials, seed):
    try:
        p, g, n, t, s = float(base_rate), float(gain), int(n_tasks), int(n_trials), int(seed)
    except (TypeError, ValueError):
        return None
    if not 0.05 <= p <= 0.9 or not 0.0 <= g <= 0.3 or p + g > 0.99:
        return None
    if n < 5 or n > 5000 or t < 50 or t > 5000:
        return None
    return p, g, n, t, s


def selection_accuracy(base_rate=0.5, gain=0.03, n_tasks=100, n_trials=1000, seed=0):
    """Return (markdown, rows). rows: list of lists [tasks, P(pick better), min detectable gain]."""
    args = _validate(base_rate, gain, n_tasks, n_trials, seed)
    if args is None:
        return ("Use a base rate 0.05-0.9, gain 0-0.3, 5-5000 tasks and 50-5000 trials.", [])
    p, g, n, t, s = args
    rng = np.random.default_rng(s)
    rows = []
    sizes = sorted({max(5, n // 4), n // 2 if n >= 10 else n, n, n * 2, n * 4, n * 10})
    for m in sizes:
        a = rng.binomial(m, p, t)
        b = rng.binomial(m, p + g, t)
        win = float(((b > a) + 0.5 * (b == a)).mean())
        md = min_detectable(m, p)
        rows.append([m, round(win * 100, 1), round(md * 100, 1)])
    cur = next(r for r in rows if r[0] == n)
    verdict = ("detectable" if g >= (min_detectable(n, p) or 1) else "below the resolution limit")
    md = (f"With **{n} tasks**, a true gain of **{g*100:.1f} points** over a {p*100:.0f}% baseline is picked "
          f"correctly **{cur[1]}%** of the time ({verdict}). Simulated, illustration only.")
    return md, rows


def evolve(base_rate=0.4, gain_per_step=0.02, n_tasks=100, generations=20, seed=0):
    """Return (markdown, rows [generation, true rate, best measured]) for a greedy keep-the-best loop."""
    args = _validate(base_rate, gain_per_step, n_tasks, 50, seed)
    try:
        gens = int(generations)
    except (TypeError, ValueError):
        gens = -1
    if args is None or gens < 1 or gens > 100:
        return ("Use a base rate 0.05-0.9, step gain 0-0.3, 5-5000 tasks and 1-100 generations.", [])
    p, g, n, _, s = args
    rng = np.random.default_rng(s)
    true, measured_best = p, rng.binomial(n, p) / n
    rows = [[0, round(true, 3), round(measured_best, 3)]]
    for k in range(1, gens + 1):
        # mutant: true quality changes by +gain_per_step or -gain_per_step with equal chance
        delta = g if rng.random() < 0.5 else -g
        cand = min(0.99, max(0.01, true + delta))
        m = rng.binomial(n, cand) / n
        if m > measured_best:
            true, measured_best = cand, m
        rows.append([k, round(true, 3), round(measured_best, 3)])
    md = (f"After {gens} generations the kept candidate's true rate is **{true*100:.1f}%** "
          f"(started at {p*100:.0f}%). Measured best: {measured_best*100:.1f}%. Simulated.")
    return md, rows


IMPL_PATTERNS = [
    {"name": "Resolution check before selection",
     "description": "Refuse to rank candidates whose score gap is below the evaluator's resolution.",
     "code": "def distinguishable(a, b, n):\n    import math\n    se = math.sqrt(a*(1-a)/n + b*(1-b)/n)\n    return abs(a - b) > 1.96 * se"},
]
