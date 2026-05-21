"""Weighted Independent Cascade opinion propagation."""
from __future__ import annotations
import numpy as np
from .graph import SocialGraph


def simulate_cascade(
    graph: SocialGraph,
    seed_ids: list[int],
    n_sims: int = 80,
    pos_opinion: float = 0.8,
    rng_seed: int = 0,
) -> dict:
    """
    Run n_sims independent cascade simulations.

    Seeds start at pos_opinion. Each step, newly activated nodes
    attempt to activate each neighbour with probability = edge weight.

    Returns mean spread and per-group coverage.
    """
    rng = np.random.default_rng(rng_seed)
    maj_ids = set(graph.group_ids("majority"))
    min_ids = set(graph.group_ids("minority"))

    total_spread = 0
    maj_spread   = 0
    min_spread   = 0

    seed_set = set(seed_ids)

    for _ in range(n_sims):
        active       = set(seed_ids)
        newly_active = set(seed_ids)

        while newly_active:
            next_new: set[int] = set()
            for node_id in newly_active:
                for nbr_id, weight in graph.neighbors(node_id):
                    if nbr_id not in active:
                        if rng.random() < weight:
                            active.add(nbr_id)
                            next_new.add(nbr_id)
            newly_active = next_new

        total_spread += len(active)
        maj_spread   += len(active & maj_ids)
        min_spread   += len(active & min_ids)

    n_maj = len(maj_ids)
    n_min = len(min_ids)

    return {
        "mean_spread":        total_spread / n_sims,
        "majority_spread":    maj_spread   / n_sims,
        "minority_spread":    min_spread   / n_sims,
        "majority_coverage":  (maj_spread / n_sims) / max(n_maj, 1),
        "minority_coverage":  (min_spread / n_sims) / max(n_min, 1),
        "fairness_gap":       abs(
            (maj_spread / n_sims) / max(n_maj, 1) -
            (min_spread / n_sims) / max(n_min, 1)
        ),
    }
