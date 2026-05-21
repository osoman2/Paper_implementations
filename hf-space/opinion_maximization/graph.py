"""Social graph with majority/minority group structure."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Node:
    id: int
    group: str        # "majority" | "minority"
    opinion: float    # initial opinion [-1, 1]
    label: str = ""


class SocialGraph:
    def __init__(self, nodes: Dict[int, Node], adj: np.ndarray):
        self.nodes = nodes          # id -> Node
        self.adj = adj              # n×n weight matrix
        self.n = len(nodes)

    def neighbors(self, node_id: int) -> List[Tuple[int, float]]:
        row = self.adj[node_id]
        return [(j, float(row[j])) for j in range(self.n) if row[j] > 0]

    def group_ids(self, group: str) -> List[int]:
        return [nid for nid, nd in self.nodes.items() if nd.group == group]

    @staticmethod
    def sample_network(
        n_majority: int = 15,
        n_minority: int = 5,
        seed: int = 42,
    ) -> "SocialGraph":
        rng = np.random.default_rng(seed)
        n_total = n_majority + n_minority
        nodes: Dict[int, Node] = {}

        for i in range(n_majority):
            nodes[i] = Node(
                id=i, group="majority",
                opinion=float(rng.uniform(-0.3, 0.3)),
                label=f"M{i}",
            )
        for i in range(n_minority):
            nid = n_majority + i
            nodes[nid] = Node(
                id=nid, group="minority",
                opinion=float(rng.uniform(-0.3, 0.3)),
                label=f"m{i}",
            )

        adj = np.zeros((n_total, n_total))

        # Intra-majority (dense)
        for i in range(n_majority):
            for j in range(i + 1, n_majority):
                if rng.random() < 0.3:
                    w = float(rng.uniform(0.1, 0.5))
                    adj[i, j] = adj[j, i] = w

        # Intra-minority (dense)
        for i in range(n_minority):
            for j in range(i + 1, n_minority):
                if rng.random() < 0.5:
                    a, b = n_majority + i, n_majority + j
                    w = float(rng.uniform(0.1, 0.4))
                    adj[a, b] = adj[b, a] = w

        # Cross-group (sparse)
        for i in range(n_majority):
            for j in range(n_minority):
                if rng.random() < 0.15:
                    a, b = i, n_majority + j
                    w = float(rng.uniform(0.05, 0.25))
                    adj[a, b] = adj[b, a] = w

        return SocialGraph(nodes=nodes, adj=adj)

    def layout_positions(self) -> Dict[int, Tuple[float, float]]:
        """Deterministic layout: minority on left, majority on right."""
        maj = sorted(self.group_ids("majority"))
        mino = sorted(self.group_ids("minority"))
        pos: Dict[int, Tuple[float, float]] = {}

        # Majority: circle around (2, 0)
        for k, nid in enumerate(maj):
            angle = 2 * np.pi * k / max(len(maj), 1)
            pos[nid] = (2.0 + 1.4 * np.cos(angle), 1.4 * np.sin(angle))

        # Minority: circle around (-1.5, 0), slightly smaller
        for k, nid in enumerate(mino):
            angle = 2 * np.pi * k / max(len(mino), 1)
            pos[nid] = (-1.5 + 0.9 * np.cos(angle), 0.9 * np.sin(angle))

        return pos
