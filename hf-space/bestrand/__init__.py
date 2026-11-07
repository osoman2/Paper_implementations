"""
BestRand: benchmark compression by (almost) random sampling — POC (Tab 54)
Source: Hugging Face Daily Papers · 2026-11-06 · arXiv 2611.00883
Score: 8/10

The paper (as summarised in the Daily Papers abstract) studies how well a small benchmark
subset reproduces full-benchmark scores. It derives an analytical approximation of the error
distribution and shows it is long-tailed: most random subsets are accurate, a few are far off.
BestRand builds on that to get stable, low-cost compression from random sampling.

This demo is an ILLUSTRATION on a simulated benchmark. It does not reproduce the paper's
algorithm or numbers; "best-of-N" below is a simple stand-in for the idea of screening random
subsets before using one.
"""

from __future__ import annotations

import math

import numpy as np

N_ITEMS = 2000
N_MODELS = 24


def build_matrix(seed: int = 0, n_items: int = N_ITEMS, n_models: int = N_MODELS) -> np.ndarray:
    """Simulated 0/1 correctness matrix (models x items) from a 1-parameter item-response model."""
    rng = np.random.default_rng(seed)
    ability = np.linspace(-1.5, 1.5, n_models)
    difficulty = rng.normal(0.0, 1.2, n_items)
    p = 1.0 / (1.0 + np.exp(-(ability[:, None] - difficulty[None, :])))
    return (rng.random((n_models, n_items)) < p).astype(np.float32)


def _subset_errors(mat: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Per-model error (subset score - full score), in accuracy points."""
    return (mat[:, idx].mean(axis=1) - mat.mean(axis=1)) * 100.0


def _prep(subset_size, n_trials, n_candidates, seed):
    try:
        k = int(subset_size)
        t = int(n_trials)
        b = int(n_candidates)
        s = int(seed)
    except (TypeError, ValueError):
        return None
    if k < 10 or k > N_ITEMS or t < 20 or t > 2000 or b < 1 or b > 200:
        return None
    return k, t, b, s


def simulate(subset_size=100, n_trials=500, n_candidates=20, seed=0):
    """Return (summary markdown, table rows, matplotlib Figure or None)."""
    args = _prep(subset_size, n_trials, n_candidates, seed)
    if args is None:
        return ("Enter a subset size between 10 and 2000, 20-2000 trials and 1-200 candidates.", [], None)
    k, t, b, s = args
    rng = np.random.default_rng(s + 1)
    mat = build_matrix(s)
    cal, held = mat[0::2], mat[1::2]  # calibration models vs held-out models

    rand_err, best_err = [], []
    for _ in range(t):
        idx = rng.choice(N_ITEMS, k, replace=False)
        rand_err.append(np.abs(_subset_errors(held, idx)).max())
        # screen b random candidates on the calibration models, keep the best one
        best_idx, best_cal = None, math.inf
        for _ in range(b):
            cand = rng.choice(N_ITEMS, k, replace=False)
            c = np.abs(_subset_errors(cal, cand)).max()
            if c < best_cal:
                best_idx, best_cal = cand, c
        best_err.append(np.abs(_subset_errors(held, best_idx)).max())
    rand_err, best_err = np.array(rand_err), np.array(best_err)

    def row(name, e):
        return [name, round(float(np.median(e)), 2), round(float(np.quantile(e, 0.95)), 2),
                round(float(np.quantile(e, 0.99)), 2), round(float(e.max()), 2)]

    rows = [row("Single random subset", rand_err), row(f"Best of {b} (screened)", best_err)]
    ratio = float(np.quantile(rand_err, 0.99) / max(np.quantile(rand_err, 0.5), 1e-9))
    text = (
        f"Simulated benchmark: {N_ITEMS} items, {N_MODELS} models (12 calibration, 12 held-out). "
        f"Subset = {k} items ({100 * k / N_ITEMS:.1f}% of the benchmark), {t} trials. "
        f"Error = worst absolute gap between subset score and full score over the held-out models, "
        f"in accuracy points. For random subsets the 99th percentile is {ratio:.1f}x the median: "
        f"the tail is what makes a single draw risky.\n\n"
        "*Illustration only: simulated data and a simplified screening rule, not the paper's algorithm.*"
    )
    fig = None
    try:
        from matplotlib.figure import Figure

        fig = Figure(figsize=(6, 3.2))
        ax = fig.subplots()
        top = float(max(rand_err.max(), best_err.max()))
        bins = np.linspace(0, top, 30)
        ax.hist(rand_err, bins=bins, alpha=0.6, label="Single random subset")
        ax.hist(best_err, bins=bins, alpha=0.6, label=f"Best of {b}")
        ax.set_xlabel("Worst held-out error (accuracy points)")
        ax.set_ylabel("Trials")
        ax.legend()
        fig.tight_layout()
    except Exception:  # plotting is optional
        fig = None
    return text, rows, fig


def subset_size_for(margin_points=2.0, accuracy=0.6, benchmark_size=2000):
    """Items needed so a random subset is within +/- margin (95%), normal approx with FPC."""
    try:
        m = float(margin_points) / 100.0
        p = float(accuracy)
        n = int(benchmark_size)
    except (TypeError, ValueError):
        return "Enter numbers."
    if not (0.0 < m < 0.5) or not (0.0 < p < 1.0) or n < 10:
        return "Use a margin between 0 and 50 points, an accuracy strictly between 0 and 1, and a benchmark of 10+ items."
    k0 = (1.96 ** 2) * p * (1 - p) / (m ** 2)  # infinite-population size
    k = k0 / (1 + (k0 - 1) / n)  # finite-population correction
    k = min(n, math.ceil(k))
    return (
        f"A random subset of about **{k}** items out of {n} ({100 * k / n:.0f}%) keeps one model's "
        f"score within +/- {margin_points} points of its full score 95% of the time (normal "
        "approximation, one model, independent items).\n\n"
        "*Illustration only: this is the textbook sampling bound, not the paper's error "
        "distribution. It does not cover rare bad subsets when comparing many models.*"
    )


FINDINGS = [
    {"name": "The problem", "body": "Running every model on a full benchmark is slow and expensive, so teams evaluate on a "
     "small subset and hope the score matches the full one."},
    {"name": "What the paper shows", "body": "The error of a random subset follows a long-tailed distribution: most subsets "
     "are close to the full score, a few are far off. The authors derive an analytical approximation of that distribution."},
    {"name": "BestRand", "body": "Because random sampling is already good with high probability, the method targets the "
     "rare bad draws to get stable, low-cost compression."},
    {"name": "What this demo does", "body": "Shows the spread of subset errors on a simulated benchmark and lets you compare a single "
     "random draw with a screened one. It is not the paper's code or results."},
]

IMPL_PATTERNS = [
    {
        "name": "Fixed-seed random subset with a recorded manifest",
        "description": "Pick the subset once, store the item ids, and reuse it for every model so scores are comparable.",
        "code": (
            "import json, random\n\n"
            "def make_subset(item_ids, k, seed=0):\n"
            "    rng = random.Random(seed)\n"
            "    return sorted(rng.sample(item_ids, k))\n\n"
            "ids = make_subset(list(range(2000)), 200)\n"
            "json.dump(ids, open('subset_manifest.json', 'w'))\n"
        ),
    },
    {
        "name": "Screen candidate subsets on reference models",
        "description": "Score a few models on the full benchmark once, then keep the candidate subset that tracks them best.",
        "code": (
            "import numpy as np\n\n"
            "def pick_subset(full_matrix, k, n_candidates=50, seed=0):\n"
            "    # full_matrix: models x items, 0/1 correctness\n"
            "    rng = np.random.default_rng(seed)\n"
            "    full = full_matrix.mean(axis=1)\n"
            "    best, best_err = None, 1e9\n"
            "    for _ in range(n_candidates):\n"
            "        idx = rng.choice(full_matrix.shape[1], k, replace=False)\n"
            "        err = np.abs(full_matrix[:, idx].mean(axis=1) - full).max()\n"
            "        if err < best_err:\n"
            "            best, best_err = idx, err\n"
            "    return best\n"
        ),
    },
]
