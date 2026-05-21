"""Greedy influence maximisation with optional fairness budget."""
from __future__ import annotations
from .graph import SocialGraph
from .cascade import simulate_cascade


def greedy_maximize(
    graph: SocialGraph,
    k: int = 3,
    fairness_budget: int = 0,
    n_sims: int = 60,
    rng_seed: int = 42,
) -> tuple[list[int], dict]:
    """
    Select k seed nodes greedily to maximise expected opinion spread.

    fairness_budget: min seeds that must come from the minority group.
    Returns (seed_ids, cascade_stats).
    """
    selected: list[int] = []
    minority_set = set(graph.group_ids("minority"))

    def marginal_gain(candidate: int) -> float:
        stats = simulate_cascade(graph, selected + [candidate], n_sims, rng_seed=rng_seed)
        return stats["mean_spread"]

    # Phase 1: fill fairness budget from minority
    forced = min(fairness_budget, k, len(minority_set))
    for _ in range(forced):
        candidates = [nid for nid in minority_set if nid not in selected]
        if not candidates:
            break
        best = max(candidates, key=marginal_gain)
        selected.append(best)

    # Phase 2: greedy fill remaining slots from all nodes
    for _ in range(k - len(selected)):
        candidates = [nid for nid in graph.nodes if nid not in selected]
        if not candidates:
            break
        best = max(candidates, key=marginal_gain)
        selected.append(best)

    stats = simulate_cascade(graph, selected, n_sims * 2, rng_seed=rng_seed + 99)
    return selected, stats
