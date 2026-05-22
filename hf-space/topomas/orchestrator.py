"""
MultiAgentOrchestrator — top-level coordinator for TopoMAS POC.

Implements the three-layer architecture from the paper:
  Layer 1: Knowledge & Resource (MaterialsKG)
  Layer 2: Multi-Agent Core    (CoreAgent → specialist agents)
  Layer 3: Application         (this orchestrator + Gradio UI)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .agents import (
    AgentResult,
    CoreAgent,
    KnowledgeAgent,
    PropertyAgent,
    SynthesisAgent,
    ValidationAgent,
)
from .knowledge_graph import MaterialsKG


@dataclass
class AgentTrace:
    """Full execution trace of one orchestrator run."""
    query: str
    task_plan: list[str] = field(default_factory=list)
    steps: list[AgentResult] = field(default_factory=list)
    final_answer: str = ""
    kg_hits: list[dict] = field(default_factory=list)
    property_data: list[dict] = field(default_factory=list)
    synthesis_candidates: list[dict] = field(default_factory=list)
    validation_verdicts: list[dict] = field(default_factory=list)

    def add_step(self, result: AgentResult) -> None:
        self.steps.append(result)

    def to_markdown_trace(self) -> str:
        lines = [f"### Agent Execution Trace\n**Query:** _{self.query}_\n"]
        lines.append(f"**Task plan:** `{'` → `'.join(self.task_plan)}`\n")
        for i, step in enumerate(self.steps, 1):
            status = "✅" if step.success else "❌"
            lines.append(f"**Step {i} — {step.agent}** {status}")
            lines.append(f"> {step.reasoning}\n")
        return "\n".join(lines)


class MultiAgentOrchestrator:
    """
    Coordinates all agents according to the task plan produced by CoreAgent.
    Implements the HITL-aware feedback loop: validated results are
    summarised back to the user and could (in a full system) be
    re-ingested into the KG.
    """

    def __init__(self) -> None:
        self.kg = MaterialsKG()
        self._core = CoreAgent()
        self._knowledge = KnowledgeAgent(self.kg)
        self._property = PropertyAgent(self.kg)
        self._synthesis = SynthesisAgent(self.kg)
        self._validation = ValidationAgent()

    # ------------------------------------------------------------------
    def run(self, query: str, top_k: int = 3) -> AgentTrace:
        trace = AgentTrace(query=query)

        # ── Layer 2, step 1: CoreAgent decomposes the query ──────────
        core_result = self._core.run(query)
        trace.add_step(core_result)
        trace.task_plan = core_result.output["task_plan"]

        # ── Execute specialist agents per task plan ───────────────────
        for task in trace.task_plan:
            if task in ("lookup", "search", "topo_query", "compare", "application"):
                result = self._knowledge.run(query, top_k=top_k)
                trace.add_step(result)
                if result.success:
                    trace.kg_hits = result.output["hits"]

            elif task == "stability":
                if not trace.kg_hits:
                    r = self._knowledge.run(query, top_k=top_k)
                    trace.add_step(r)
                    trace.kg_hits = r.output.get("hits", [])
                prop_result = self._property.run(trace.kg_hits)
                trace.add_step(prop_result)
                trace.property_data = prop_result.output.get("property_analysis", [])

            elif task == "synthesis":
                synth_result = self._synthesis.run(query)
                trace.add_step(synth_result)
                trace.synthesis_candidates = synth_result.output.get("candidates", [])

            elif task == "validation":
                if not trace.property_data and trace.kg_hits:
                    prop_result = self._property.run(trace.kg_hits)
                    trace.add_step(prop_result)
                    trace.property_data = prop_result.output.get("property_analysis", [])
                val_result = self._validation.run(trace.property_data)
                trace.add_step(val_result)
                trace.validation_verdicts = val_result.output.get("verdicts", [])

        # ── Always run property + validation if we have KG hits ───────
        if trace.kg_hits and not trace.property_data:
            prop_result = self._property.run(trace.kg_hits)
            trace.add_step(prop_result)
            trace.property_data = prop_result.output.get("property_analysis", [])

        if trace.property_data and not trace.validation_verdicts:
            val_result = self._validation.run(trace.property_data)
            trace.add_step(val_result)
            trace.validation_verdicts = val_result.output.get("verdicts", [])

        # ── Build final natural-language answer ───────────────────────
        trace.final_answer = _compose_answer(trace)
        return trace


# ---------------------------------------------------------------------------
# Answer composer — aggregates agent outputs into readable text
# ---------------------------------------------------------------------------
def _compose_answer(trace: AgentTrace) -> str:
    parts: list[str] = []

    if trace.kg_hits:
        parts.append("**Materials retrieved from knowledge graph:**\n")
        for h in trace.kg_hits:
            topo_flag = "🔬 Topological" if h["is_topological"] else "⚪ Trivial"
            parts.append(
                f"- **{h['formula']}** ({h['name']}) — "
                f"{h['topo_class']} | {topo_flag} | "
                f"Gap: {h['band_gap_eV']} eV | {h['stability'].capitalize()}\n"
                f"  _{h['notes']}_"
            )

    if trace.validation_verdicts:
        parts.append("\n**Validation results:**\n")
        for v in trace.validation_verdicts:
            emoji = {"VALID": "✅", "VALID (trivial)": "✅", "INCONSISTENT — claimed topological but Z₂=0": "⚠️"}.get(v.get("verdict", ""), "❓")
            parts.append(
                f"- {emoji} **{v.get('formula', '?')}** — {v.get('verdict', 'UNKNOWN')}"
            )

    if trace.synthesis_candidates:
        parts.append("\n**Hypothetical material candidates (SynthesisAgent):**\n")
        for c in trace.synthesis_candidates:
            parts.append(
                f"- **{c['formula']}** ({c['type']}) — "
                f"{c['predicted_topo_class']} | "
                f"Predicted gap: {c['predicted_band_gap_eV']} eV | {c['predicted_stability']}\n"
                f"  _{c['rationale']}_"
            )

    if not parts:
        parts.append("No results found. Try a different query (e.g. 'Bi2Se3', 'weyl semimetal', 'topological insulator applications').")

    return "\n".join(parts)
