"""
In-memory knowledge graph of topological materials.
Mirrors the knowledge-and-resource layer in TopoMAS (Zhang et al.).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Material:
    formula: str
    name: str
    topo_class: str          # TI, TSM, NI, HOTI …
    band_gap_eV: float
    is_topological: bool
    stability: str           # stable / metastable / unstable
    space_group: str
    applications: list[str]
    properties: dict[str, float] = field(default_factory=dict)
    notes: str = ""


# ---------------------------------------------------------------------------
# Curated knowledge base — representative topological materials
# ---------------------------------------------------------------------------
_MATERIALS: list[Material] = [
    Material(
        formula="Bi2Se3",
        name="Bismuth selenide",
        topo_class="Topological Insulator (TI)",
        band_gap_eV=0.30,
        is_topological=True,
        stability="stable",
        space_group="R-3m",
        applications=["spintronics", "quantum computing", "topological qubits"],
        properties={"formation_energy": -0.52, "density": 7.68, "z2_invariant": 1.0},
        notes="Prototypical 3-D TI; surface states protected by time-reversal symmetry.",
    ),
    Material(
        formula="WTe2",
        name="Tungsten ditelluride",
        topo_class="Type-II Weyl Semimetal (TSM)",
        band_gap_eV=0.0,
        is_topological=True,
        stability="stable",
        space_group="Pmn21",
        applications=["non-linear Hall effect", "piezoelectricity", "magnetoresistance"],
        properties={"formation_energy": -0.84, "density": 9.43, "chern_number": 2.0},
        notes="Large, non-saturating magnetoresistance; first predicted Type-II Weyl semimetal.",
    ),
    Material(
        formula="Bi2Te3",
        name="Bismuth telluride",
        topo_class="Topological Insulator (TI)",
        band_gap_eV=0.15,
        is_topological=True,
        stability="stable",
        space_group="R-3m",
        applications=["thermoelectrics", "spintronics", "quantum devices"],
        properties={"formation_energy": -0.46, "density": 7.86, "z2_invariant": 1.0},
        notes="Excellent thermoelectric; topological surface states coexist with bulk conduction.",
    ),
    Material(
        formula="MnBi2Te4",
        name="Manganese bismuth telluride",
        topo_class="Magnetic Topological Insulator (MTI)",
        band_gap_eV=0.20,
        is_topological=True,
        stability="stable",
        space_group="R-3m",
        applications=["quantum anomalous Hall effect", "axion insulators", "spintronics"],
        properties={"formation_energy": -0.61, "density": 8.15, "z2_invariant": 1.0},
        notes="First intrinsic magnetic TI; hosts quantum anomalous Hall state below 6.5 K.",
    ),
    Material(
        formula="TaAs",
        name="Tantalum arsenide",
        topo_class="Weyl Semimetal (WSM)",
        band_gap_eV=0.0,
        is_topological=True,
        stability="stable",
        space_group="I41md",
        applications=["Fermi arc surface states", "non-linear optics", "photovoltaics"],
        properties={"formation_energy": -0.71, "density": 11.07, "chern_number": 1.0},
        notes="First experimentally confirmed Weyl semimetal; 12 pairs of Weyl nodes.",
    ),
    Material(
        formula="Sb2Te3",
        name="Antimony telluride",
        topo_class="Topological Insulator (TI)",
        band_gap_eV=0.21,
        is_topological=True,
        stability="stable",
        space_group="R-3m",
        applications=["phase-change memory", "thermoelectrics", "IR sensors"],
        properties={"formation_energy": -0.34, "density": 6.50, "z2_invariant": 1.0},
        notes="Smaller band gap than Bi2Se3; used in PCM applications alongside topological properties.",
    ),
    Material(
        formula="HgTe",
        name="Mercury telluride",
        topo_class="Topological Insulator (TI)",
        band_gap_eV=0.0,
        is_topological=True,
        stability="stable",
        space_group="F-43m",
        applications=["quantum wells", "infrared detectors", "topological edge states"],
        properties={"formation_energy": -0.22, "density": 8.12, "z2_invariant": 1.0},
        notes="Zero-gap semiconductor; becomes TI in quantum-well geometry (BHZ model).",
    ),
    Material(
        formula="Na3Bi",
        name="Sodium bismuthide",
        topo_class="Dirac Semimetal (DSM)",
        band_gap_eV=0.0,
        is_topological=True,
        stability="metastable",
        space_group="P63/mmc",
        applications=["Dirac surface states", "topological electronics"],
        properties={"formation_energy": -0.18, "density": 3.21},
        notes="3-D Dirac semimetal; air-sensitive. Hosts linear band crossings.",
    ),
    Material(
        formula="Si",
        name="Silicon",
        topo_class="Normal Insulator (NI)",
        band_gap_eV=1.12,
        is_topological=False,
        stability="stable",
        space_group="Fd-3m",
        applications=["semiconductors", "solar cells", "CMOS"],
        properties={"formation_energy": 0.0, "density": 2.33, "z2_invariant": 0.0},
        notes="Topologically trivial benchmark; no protected surface states.",
    ),
    Material(
        formula="GaAs",
        name="Gallium arsenide",
        topo_class="Normal Semiconductor (NS)",
        band_gap_eV=1.42,
        is_topological=False,
        stability="stable",
        space_group="F-43m",
        applications=["LEDs", "solar cells", "high-speed electronics"],
        properties={"formation_energy": -0.35, "density": 5.32, "z2_invariant": 0.0},
        notes="High electron mobility; topologically trivial but important III-V semiconductor.",
    ),
]

_INDEX: dict[str, Material] = {m.formula.lower(): m for m in _MATERIALS}


class MaterialsKG:
    """Simple in-memory knowledge graph over topological materials."""

    def __init__(self) -> None:
        self._data = {m.formula: m for m in _MATERIALS}

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> list[Material]:
        """Keyword-based retrieval over formula, name, class, applications, notes."""
        tokens = query.lower().split()
        scored: list[tuple[float, Material]] = []
        for mat in self._data.values():
            blob = " ".join([
                mat.formula, mat.name, mat.topo_class,
                mat.space_group, mat.notes,
                " ".join(mat.applications),
            ]).lower()
            score = sum(blob.count(t) for t in tokens)
            if score:
                scored.append((score, mat))
        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored[:top_k]]

    def get(self, formula: str) -> Optional[Material]:
        return self._data.get(formula)

    def topological_only(self) -> list[Material]:
        return [m for m in self._data.values() if m.is_topological]

    def by_class(self, topo_class: str) -> list[Material]:
        tc = topo_class.lower()
        return [m for m in self._data.values() if tc in m.topo_class.lower()]

    def all(self) -> list[Material]:
        return list(self._data.values())
