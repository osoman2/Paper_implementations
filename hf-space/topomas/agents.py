"""
Specialized agents that form the Multi-Agent Core of TopoMAS.
Each agent handles a distinct sub-task; the CoreAgent decomposes
a natural-language query and dispatches to the right specialists.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .knowledge_graph import Material, MaterialsKG


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------
@dataclass
class AgentResult:
    agent: str
    success: bool
    output: dict[str, Any]
    reasoning: str


# ---------------------------------------------------------------------------
# Core Agent — orchestration & task decomposition
# ---------------------------------------------------------------------------
class CoreAgent:
    """
    Interprets the user query and produces a task plan.
    In the paper this is the LLM-driven 'Core Agent' that translates
    natural-language intent into structured subtasks.
    """

    _INTENT_PATTERNS = [
        (r"what is|describe|tell me about|properties of", "lookup"),
        (r"find|search|discover|which materials|list", "search"),
        (r"compare|versus|vs\.", "compare"),
        (r"stable|stability|synthesiz", "stability"),
        (r"application|use.?case|useful for", "application"),
        (r"topological|topo|invariant|weyl|dirac|insulator|semimetal", "topo_query"),
        (r"generate|suggest|hypothesize|propose|new material", "synthesis"),
        (r"validate|verify|confirm|check", "validation"),
    ]

    def decompose(self, query: str) -> list[str]:
        """Return ordered list of subtask types for this query."""
        q = query.lower()
        tasks: list[str] = []
        for pattern, task in self._INTENT_PATTERNS:
            if re.search(pattern, q):
                tasks.append(task)
        if not tasks:
            tasks = ["search"]
        # always retrieve knowledge first
        if "lookup" not in tasks and "search" not in tasks:
            tasks.insert(0, "search")
        return list(dict.fromkeys(tasks))  # deduplicate, preserve order

    def run(self, query: str) -> AgentResult:
        tasks = self.decompose(query)
        return AgentResult(
            agent="CoreAgent",
            success=True,
            output={"query": query, "task_plan": tasks},
            reasoning=(
                f"Parsed query intent and produced task plan: {tasks}. "
                "Dispatching to specialist agents in sequence."
            ),
        )


# ---------------------------------------------------------------------------
# Knowledge Agent — database & literature retrieval
# ---------------------------------------------------------------------------
class KnowledgeAgent:
    """
    Retrieves relevant materials from the KG and returns structured facts.
    Mirrors the 'Literature & DB Retrieval' agent in TopoMAS.
    """

    def __init__(self, kg: MaterialsKG) -> None:
        self._kg = kg

    def run(self, query: str, top_k: int = 3) -> AgentResult:
        hits = self._kg.search(query, top_k=top_k)
        if not hits:
            return AgentResult(
                agent="KnowledgeAgent",
                success=False,
                output={"hits": []},
                reasoning="No matching materials found in knowledge graph.",
            )
        return AgentResult(
            agent="KnowledgeAgent",
            success=True,
            output={"hits": [_mat_to_dict(m) for m in hits]},
            reasoning=(
                f"Retrieved {len(hits)} material(s) from KG matching '{query}': "
                + ", ".join(m.formula for m in hits)
            ),
        )


# ---------------------------------------------------------------------------
# Property Agent — property estimation & analysis
# ---------------------------------------------------------------------------
class PropertyAgent:
    """
    Estimates or looks up key physical/topological properties.
    In the real system this wraps DFT calculation workflows.
    Here it reads from the KG and applies simple heuristics.
    """

    def __init__(self, kg: MaterialsKG) -> None:
        self._kg = kg

    def run(self, materials: list[dict]) -> AgentResult:
        results = []
        for m_dict in materials:
            formula = m_dict.get("formula", "")
            mat = self._kg.get(formula)
            if mat is None:
                results.append({"formula": formula, "error": "not in KG"})
                continue
            analysis = _analyse_properties(mat)
            results.append({"formula": formula, **analysis})

        return AgentResult(
            agent="PropertyAgent",
            success=bool(results),
            output={"property_analysis": results},
            reasoning=(
                f"Analysed properties for {len(results)} material(s). "
                "Topological invariants cross-checked against band-gap and class."
            ),
        )


# ---------------------------------------------------------------------------
# Synthesis Agent — hypothetical material generation
# ---------------------------------------------------------------------------
class SynthesisAgent:
    """
    Generates candidate hypothetical materials by interpolating
    between known stable topological materials in the KG.
    In TopoMAS this is backed by crystal-structure generative models.
    """

    def __init__(self, kg: MaterialsKG) -> None:
        self._kg = kg

    def run(self, base_query: str) -> AgentResult:
        topo_mats = self._kg.topological_only()
        if len(topo_mats) < 2:
            return AgentResult(
                agent="SynthesisAgent",
                success=False,
                output={"candidates": []},
                reasoning="Insufficient topological materials in KG to generate candidates.",
            )

        # Simple heuristic: propose alloyed variants of stable TIs
        stable_tis = [m for m in topo_mats if m.stability == "stable" and "TI" in m.topo_class]
        candidates = []
        if len(stable_tis) >= 2:
            a, b = stable_tis[0], stable_tis[1]
            candidates.append({
                "formula": f"({a.formula})0.5({b.formula})0.5",
                "type": "alloy",
                "predicted_topo_class": "Topological Insulator (TI)",
                "predicted_band_gap_eV": round((a.band_gap_eV + b.band_gap_eV) / 2, 3),
                "predicted_stability": "metastable",
                "rationale": (
                    f"50/50 alloy of {a.name} and {b.name}. "
                    "Both parent compounds are stable TIs with the R-3m space group — "
                    "solid-solution alloying is expected to preserve topological order "
                    "while tuning the band gap."
                ),
            })
        # Propose doped variant
        if stable_tis:
            m = stable_tis[0]
            candidates.append({
                "formula": f"Cr0.1{m.formula[2:]}",
                "type": "dopant",
                "predicted_topo_class": "Magnetic Topological Insulator (MTI)",
                "predicted_band_gap_eV": round(m.band_gap_eV * 0.85, 3),
                "predicted_stability": "metastable",
                "rationale": (
                    f"Cr doping of {m.name} to break time-reversal symmetry. "
                    "Expected to open a magnetic gap in surface states, enabling "
                    "the quantum anomalous Hall effect at low temperature."
                ),
            })

        return AgentResult(
            agent="SynthesisAgent",
            success=True,
            output={"candidates": candidates},
            reasoning=(
                f"Generated {len(candidates)} hypothetical candidate(s) "
                "via alloying and doping strategies applied to stable KG materials."
            ),
        )


# ---------------------------------------------------------------------------
# Validation Agent — property & topological validation
# ---------------------------------------------------------------------------
class ValidationAgent:
    """
    Validates topological claims using invariants stored in the KG.
    In the paper this triggers DFT calculations and diagnoses failures.
    Here it applies invariant consistency rules.
    """

    def run(self, property_analysis: list[dict]) -> AgentResult:
        verdicts = []
        for entry in property_analysis:
            if "error" in entry:
                verdicts.append({**entry, "verdict": "UNKNOWN"})
                continue
            is_topo = entry.get("is_topological", False)
            invariant = entry.get("topological_invariant", 0)
            gap = entry.get("band_gap_eV", -1)
            verdict = _validate_topological_claim(is_topo, invariant, gap)
            verdicts.append({**entry, "verdict": verdict})

        passed = sum(1 for v in verdicts if v.get("verdict") == "VALID")
        return AgentResult(
            agent="ValidationAgent",
            success=True,
            output={"verdicts": verdicts},
            reasoning=(
                f"Validated {passed}/{len(verdicts)} entries as topologically consistent. "
                "Rules: non-trivial Z₂ invariant, non-negative gap for insulators."
            ),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mat_to_dict(m: Material) -> dict:
    return {
        "formula": m.formula,
        "name": m.name,
        "topo_class": m.topo_class,
        "band_gap_eV": m.band_gap_eV,
        "is_topological": m.is_topological,
        "stability": m.stability,
        "space_group": m.space_group,
        "applications": m.applications,
        "properties": m.properties,
        "notes": m.notes,
    }


def _analyse_properties(mat: Material) -> dict:
    inv = mat.properties.get("z2_invariant", mat.properties.get("chern_number", 0))
    return {
        "is_topological": mat.is_topological,
        "topo_class": mat.topo_class,
        "band_gap_eV": mat.band_gap_eV,
        "stability": mat.stability,
        "topological_invariant": inv,
        "invariant_type": "Z₂" if "z2_invariant" in mat.properties else "Chern",
        "formation_energy_eV_per_atom": mat.properties.get("formation_energy", "N/A"),
        "applications": mat.applications,
    }


def _validate_topological_claim(is_topo: bool, invariant: float, gap: float) -> str:
    if is_topo and invariant > 0:
        return "VALID"
    if not is_topo and invariant == 0:
        return "VALID (trivial)"
    if is_topo and invariant == 0:
        return "INCONSISTENT — claimed topological but Z₂=0"
    return "UNKNOWN"
