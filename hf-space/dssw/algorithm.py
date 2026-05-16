"""
Dynamic Sub-Sequence Warping (DSSW)
POC based on: "Dynamic Sub-Sequence Warping: A Representation-Based
Similarity Measure for Long Time Series" (score 9/10).

Key idea: instead of aligning individual points like DTW, DSSW first
segments each series into overlapping sub-sequences and then runs a
DTW-style DP over those sub-sequence representations. This amortises
the O(n*m) DTW cost, making it tractable for long series, while
retaining sensitivity to local shape patterns.

Retrieval relevance: useful for comparing embedding-drift trajectories,
agent-state histories, or sequential document relevance scores — any
scenario where two "runs" need to be matched despite time-scale shifts.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Sub-sequence extraction
# ---------------------------------------------------------------------------

def extract_subsequences(ts: np.ndarray, length: int, stride: int) -> np.ndarray:
    """Return shape (n_subs, length) array of overlapping windows."""
    ts = np.asarray(ts, dtype=float)
    n = len(ts)
    if length > n:
        raise ValueError(f"sub-sequence length {length} > series length {n}")
    starts = range(0, n - length + 1, stride)
    return np.stack([ts[s : s + length] for s in starts])


# ---------------------------------------------------------------------------
# Pairwise sub-sequence distance matrix
# ---------------------------------------------------------------------------

def _sub_dist_matrix(subs1: np.ndarray, subs2: np.ndarray) -> np.ndarray:
    """Euclidean distance between every pair of sub-sequences (n×m)."""
    # subs1: (n, L), subs2: (m, L)
    diff = subs1[:, None, :] - subs2[None, :, :]   # (n, m, L)
    return np.sqrt((diff ** 2).sum(axis=-1))        # (n, m)


# ---------------------------------------------------------------------------
# DSSW core
# ---------------------------------------------------------------------------

def dssw_distance(
    ts1: list | np.ndarray,
    ts2: list | np.ndarray,
    subseq_len: int = 10,
    stride: int = 5,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    distance : float
        Normalised DSSW distance (lower = more similar).
    D : np.ndarray
        Raw sub-sequence distance matrix (n×m).
    path : np.ndarray
        Optimal warping path as array of (i, j) index pairs.
    """
    subs1 = extract_subsequences(np.asarray(ts1), subseq_len, stride)
    subs2 = extract_subsequences(np.asarray(ts2), subseq_len, stride)
    n, m  = len(subs1), len(subs2)

    D = _sub_dist_matrix(subs1, subs2)

    # DP (same recurrence as DTW, applied at sub-sequence level)
    INF = np.inf
    dp  = np.full((n + 1, m + 1), INF)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost    = D[i - 1, j - 1]
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    # Traceback
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        choices = {
            (i - 1, j - 1): dp[i - 1, j - 1],
            (i - 1, j):     dp[i - 1, j],
            (i, j - 1):     dp[i, j - 1],
        }
        i, j = min(choices, key=choices.get)
    path.reverse()

    dist = dp[n, m] / (n + m)
    return dist, D, np.array(path)


# ---------------------------------------------------------------------------
# Standard DTW (point-level) for comparison
# ---------------------------------------------------------------------------

def dtw_distance(ts1: list | np.ndarray, ts2: list | np.ndarray) -> float:
    """O(n·m) DTW with Euclidean point cost."""
    a = np.asarray(ts1, dtype=float)
    b = np.asarray(ts2, dtype=float)
    n, m = len(a), len(b)
    dp   = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost    = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return dp[n, m]


# ---------------------------------------------------------------------------
# Euclidean distance (same-length baseline)
# ---------------------------------------------------------------------------

def euclidean_distance(ts1: list | np.ndarray, ts2: list | np.ndarray) -> float:
    a = np.asarray(ts1, dtype=float)
    b = np.asarray(ts2, dtype=float)
    min_len = min(len(a), len(b))
    return float(np.sqrt(np.sum((a[:min_len] - b[:min_len]) ** 2)))
