"""
Shopify Self-Improving AI Pipeline (Production Flywheel) Demo

Based on: AlphaSignal deep-dive · 2026-09-06 · news@alphasignal.ai
"What Developers Can Learn From Shopify's Self-Improving AI Pipeline"
  - Shopify fine-tuned a Qwen3.5 0.8B model to outperform GPT-5.6 Sol xhigh
    on a narrow buyer-profile generation task
  - Key mechanism: production flywheel — failures become training data via
    LLM-judge critique + repair → SFT + GRPO retraining
  - Cost: $27M/yr (frontier) → $1M/yr (specialized), 38% latency reduction,
    16% throughput increase, 14% fewer GPUs

Score: 8/10 — concrete production case with implementable flywheel pattern,
LLM-judge calibration methodology, and quantified cost/latency metrics.
Directly applicable to any high-volume, bounded LLM task.

NOTE: This module uses deterministic simulations. In production, replace
with actual LLM inference, real evaluation rubrics, and live training loops.
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProductionConversation:
    id: str
    task: str
    agent_output: str
    evaluator_score: float      # 0–1 from automated evaluator
    criteria: dict[str, float]  # rubric dimension → score
    is_failure: bool
    failure_reason: str | None


@dataclass
class RepairRecord:
    original: ProductionConversation
    frontier_critique: str
    repair_instruction: str
    repaired_output: str
    repaired_score: float
    became_training_data: bool


@dataclass
class JudgeCalibration:
    criterion: str
    judge_scores: list[float]   # one per LLM judge (4 judges)
    human_score: float
    kappa: float
    consensus: bool             # all 4 judges agree
    label_accepted: bool        # accepted only if all 4 agree


@dataclass
class CostComparison:
    task: str
    daily_requests: int
    frontier_cost_per_1k: float     # USD per 1000 requests
    specialized_cost_per_1k: float
    frontier_annual_usd: int
    specialized_annual_usd: int
    latency_reduction_pct: float
    throughput_increase_pct: float
    gpu_reduction_pct: float
    prompt_token_reduction_pct: float


# ---------------------------------------------------------------------------
# Example production conversations (simulated)
# ---------------------------------------------------------------------------

EXAMPLE_CONVERSATIONS: list[ProductionConversation] = [
    ProductionConversation(
        id="conv_001",
        task="Generate buyer profile: merchant sells vintage watches, customer browsed 3 Rolex listings",
        agent_output=(
            "Customer profile: High-income professional interested in luxury timepieces. "
            "Likely 35–55, values heritage and craftsmanship. "
            "Recommend: Rolex Submariner, Omega Seamaster, Patek Philippe Calatrava."
        ),
        evaluator_score=0.91,
        criteria={"completeness": 0.90, "execution": 0.95, "response_quality": 0.88, "safety": 1.0},
        is_failure=False,
        failure_reason=None,
    ),
    ProductionConversation(
        id="conv_002",
        task="Generate buyer profile: merchant sells handmade pottery, customer asked about 'bowls'",
        agent_output=(
            "Customer profile: Interested in kitchen items. "
            "Recommend: ceramic sets, mixing bowls, salad bowls."
        ),
        evaluator_score=0.42,
        criteria={"completeness": 0.35, "execution": 0.50, "response_quality": 0.40, "safety": 1.0},
        is_failure=True,
        failure_reason="Profile too generic — ignores artisanal/handmade context. No lifestyle inference. Recommendations don't match shop style.",
    ),
    ProductionConversation(
        id="conv_003",
        task="Generate buyer profile: merchant sells children's educational toys, customer browsed STEM kits for age 8",
        agent_output=(
            "Customer profile: Parent seeking educational enrichment for child aged ~8. "
            "Values learning outcomes over entertainment. Budget: mid-range ($30–80). "
            "Recommend: robotics starter kit, coding board games, science experiment sets."
        ),
        evaluator_score=0.87,
        criteria={"completeness": 0.90, "execution": 0.85, "response_quality": 0.85, "safety": 1.0},
        is_failure=False,
        failure_reason=None,
    ),
    ProductionConversation(
        id="conv_004",
        task="Generate buyer profile: merchant sells hiking gear, customer asked if store 'has jackets for Patagonia trip'",
        agent_output=(
            "Customer profile: Outdoor enthusiast. "
            "Recommend: fleece jackets, waterproof shells."
        ),
        evaluator_score=0.31,
        criteria={"completeness": 0.20, "execution": 0.40, "response_quality": 0.30, "safety": 1.0},
        is_failure=True,
        failure_reason="Missed trip-specific context (Patagonia = demanding conditions). Profile too shallow. Missed upsell opportunity for layering system, technical specifications.",
    ),
    ProductionConversation(
        id="conv_005",
        task="Reject impossible segmentation request: 'Show me customers who bought from us in 1990'",
        agent_output=(
            "Here are customers who purchased in 1990: [generating list...]"
        ),
        evaluator_score=0.0,
        criteria={"completeness": 0.0, "execution": 0.0, "response_quality": 0.0, "safety": 0.5},
        is_failure=True,
        failure_reason="Should reject — store has no 1990 data. Critical failure: hallucinating historical records.",
    ),
]

CONVERSATION_LABELS = [f"[{'✅' if not c.is_failure else '❌'} {c.score_label()}] {c.task[:60]}..."
                       if hasattr(ProductionConversation, 'score_label') else c.id
                       for c in EXAMPLE_CONVERSATIONS]


def get_conversation_labels() -> list[str]:
    return [
        f"{'✅' if not c.is_failure else '❌'} [{c.evaluator_score:.0%}] {c.task[:55]}..."
        for c in EXAMPLE_CONVERSATIONS
    ]


def get_conversation(label: str) -> ProductionConversation | None:
    labels = get_conversation_labels()
    for i, l in enumerate(labels):
        if l == label:
            return EXAMPLE_CONVERSATIONS[i]
    return None


# ---------------------------------------------------------------------------
# Repair simulations (frontier model critique + repair)
# ---------------------------------------------------------------------------

REPAIRS: dict[str, RepairRecord] = {
    "conv_002": RepairRecord(
        original=EXAMPLE_CONVERSATIONS[1],
        frontier_critique=(
            "The output fails to leverage the artisanal context of the shop. "
            "'Handmade pottery' signals a customer who values craft, individuality, and maker stories — "
            "not commodity kitchen items. The profile should infer: gift-buyer or collector mindset, "
            "appreciation for unique one-of-a-kind pieces, likely browsing for a specific occasion. "
            "Recommendations should match the shop's aesthetic, not a generic kitchenware category."
        ),
        repair_instruction=(
            "Rewrite the profile to: (1) infer lifestyle and purchase motivation from the artisanal context, "
            "(2) identify occasion (gift, home décor, personal collection), "
            "(3) recommend pieces that match the shop's handmade aesthetic."
        ),
        repaired_output=(
            "Customer profile: Gift-seeker or collector drawn to artisan craft and tactile authenticity. "
            "Likely values unique, one-of-a-kind pieces over mass-produced alternatives — browsing 'bowls' "
            "may indicate looking for a housewarming or wedding gift. "
            "Recommend: showcasing the maker's story, highlighting glaze techniques, suggesting gift packaging. "
            "Upsell: complementary pieces (pitcher, mugs) to complete a matching set."
        ),
        repaired_score=0.89,
        became_training_data=True,
    ),
    "conv_004": RepairRecord(
        original=EXAMPLE_CONVERSATIONS[3],
        frontier_critique=(
            "The Patagonia trip mention is a high-signal context cue: demanding weather, remote terrain, "
            "multi-day hiking. The profile ignored this entirely. A proper profile should infer: "
            "experienced hiker, prioritizes technical performance over price, likely needs a complete "
            "layering system (base/mid/shell), not just 'jackets'. Missed upsell value is high."
        ),
        repair_instruction=(
            "Rewrite with: (1) infer experience level and technical requirements from trip destination, "
            "(2) build a complete layering recommendation, (3) highlight waterproofing ratings and weight."
        ),
        repaired_output=(
            "Customer profile: Experienced hiker planning a demanding multi-day trip to Patagonia — "
            "expects wind, rain, and cold. Prioritizes technical performance (waterproofing, packability, weight) "
            "over aesthetics. Budget: likely $150–400 for outerwear. "
            "Recommend: hard-shell jacket (10,000mm+ waterproof rating), insulating mid-layer, moisture-wicking base. "
            "Upsell: trekking poles, gaiters, dry bags — all relevant for Patagonia conditions."
        ),
        repaired_score=0.93,
        became_training_data=True,
    ),
    "conv_005": RepairRecord(
        original=EXAMPLE_CONVERSATIONS[4],
        frontier_critique=(
            "Critical failure: the model attempted to generate data for a request that is logically impossible "
            "(store founded after 1990). The correct behavior is to detect the impossible constraint and "
            "gracefully reject with an explanation. The model should have been trained on refusal examples "
            "but none existed in the production dataset."
        ),
        repair_instruction=(
            "Generate a graceful rejection: explain that the store's data only goes back to [founding year], "
            "suggest what is possible instead (earliest available data), and offer to run the query on the "
            "available range."
        ),
        repaired_output=(
            "I can't retrieve customers from 1990 — our store's records begin in 2018. "
            "I can show you your earliest customers (2018–2019) or run this analysis for any date range "
            "since then. Which would you prefer?"
        ),
        repaired_score=0.97,
        became_training_data=True,
    ),
}


def get_repair(conv_id: str) -> RepairRecord | None:
    return REPAIRS.get(conv_id)


# ---------------------------------------------------------------------------
# LLM judge calibration examples
# ---------------------------------------------------------------------------

JUDGE_CALIBRATIONS: list[JudgeCalibration] = [
    JudgeCalibration(
        criterion="Completeness",
        judge_scores=[0.85, 0.80, 0.90, 0.85],
        human_score=0.85,
        kappa=0.82,
        consensus=True,
        label_accepted=True,
    ),
    JudgeCalibration(
        criterion="Execution",
        judge_scores=[0.70, 0.90, 0.75, 0.80],
        human_score=0.78,
        kappa=0.54,
        consensus=False,
        label_accepted=False,
    ),
    JudgeCalibration(
        criterion="Response Quality",
        judge_scores=[0.60, 0.65, 0.60, 0.62],
        human_score=0.62,
        kappa=0.78,
        consensus=True,
        label_accepted=True,
    ),
    JudgeCalibration(
        criterion="Safety",
        judge_scores=[1.0, 1.0, 1.0, 1.0],
        human_score=1.0,
        kappa=1.0,
        consensus=True,
        label_accepted=True,
    ),
    JudgeCalibration(
        criterion="Refusal Accuracy (impossible requests)",
        judge_scores=[0.0, 0.5, 0.0, 1.0],
        human_score=0.0,
        kappa=0.12,
        consensus=False,
        label_accepted=False,
    ),
]


def kappa_label(k: float) -> str:
    if k >= 0.80:
        return f"✅ Strong ({k:.2f})"
    if k >= 0.60:
        return f"🟡 Moderate ({k:.2f})"
    return f"❌ Poor ({k:.2f})"


def format_judge_table() -> str:
    rows = [
        "| Criterion | Judge 1 | Judge 2 | Judge 3 | Judge 4 | Human | κ (kappa) | Accepted? |",
        "|-----------|---------|---------|---------|---------|-------|-----------|-----------|",
    ]
    for jc in JUDGE_CALIBRATIONS:
        j_scores = " | ".join(f"{s:.0%}" for s in jc.judge_scores)
        accepted = "✅ Yes" if jc.label_accepted else "❌ No — sent to annotator"
        rows.append(
            f"| **{jc.criterion}** | {j_scores} | {jc.human_score:.0%} "
            f"| {kappa_label(jc.kappa)} | {accepted} |"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Cost comparison data
# ---------------------------------------------------------------------------

COST_COMPARISON = CostComparison(
    task="Buyer profile generation (Shopify Sidekick)",
    daily_requests=2_000_000,
    frontier_cost_per_1k=37.0,
    specialized_cost_per_1k=1.37,
    frontier_annual_usd=27_010_000,
    specialized_annual_usd=1_000_000,
    latency_reduction_pct=38.0,
    throughput_increase_pct=16.0,
    gpu_reduction_pct=14.0,
    prompt_token_reduction_pct=75.0,
)

GRAPHQL_COMPARISON = CostComparison(
    task="GraphQL agent (Shopify Admin)",
    daily_requests=2_880_000,   # 2000 req/min × 60 × 24
    frontier_cost_per_1k=9.38,
    specialized_cost_per_1k=0.35,
    frontier_annual_usd=9_870_000,
    specialized_annual_usd=370_000,
    latency_reduction_pct=19.0,    # TTFT reduction
    throughput_increase_pct=16.0,
    gpu_reduction_pct=14.0,
    prompt_token_reduction_pct=75.0,  # 6000 → 1500 tokens
)


def format_cost_table(comp: CostComparison) -> str:
    savings = comp.frontier_annual_usd - comp.specialized_annual_usd
    savings_pct = savings / comp.frontier_annual_usd * 100
    rows = [
        f"| Metric | Frontier Model | Specialized 0.8B | Delta |",
        f"|--------|---------------|------------------|-------|",
        f"| Annual cost | **${comp.frontier_annual_usd:,.0f}** | **${comp.specialized_annual_usd:,.0f}** | 💰 **-${savings:,.0f} ({savings_pct:.0f}% savings)** |",
        f"| Cost per 1K requests | ${comp.frontier_cost_per_1k:.2f} | ${comp.specialized_cost_per_1k:.2f} | {comp.frontier_cost_per_1k / comp.specialized_cost_per_1k:.0f}× cheaper |",
        f"| Latency (E2E) | baseline | -{comp.latency_reduction_pct:.0f}% | ⚡ faster |",
        f"| Throughput | baseline | +{comp.throughput_increase_pct:.0f}% | ↑ more requests |",
        f"| GPU count | baseline | -{comp.gpu_reduction_pct:.0f}% | 🖥️ fewer servers |",
        f"| Prompt tokens | ~6,000 | ~1,500 | -{comp.prompt_token_reduction_pct:.0f}% |",
    ]
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Flywheel stages
# ---------------------------------------------------------------------------

FLYWHEEL_STAGES: list[dict] = [
    {
        "step": 1,
        "icon": "🏭",
        "title": "Production: Frontier Model Serves Traffic",
        "detail": (
            "Deploy a frontier model (e.g., GPT-5.6, Claude Opus) to handle production requests. "
            "This phase prioritizes flexibility: prompts, tools, and logic can be iterated quickly "
            "without retraining. The frontier model generates the ground-truth distribution of outputs "
            "and surfaces real-world failure modes that synthetic data cannot anticipate."
        ),
        "code_pattern": "response = frontier_model.complete(system_prompt, user_turn)",
    },
    {
        "step": 2,
        "icon": "⚖️",
        "title": "Evaluate: LLM Judges Score Every Conversation",
        "detail": (
            "Automated LLM evaluators score each conversation against a rubric (completeness, execution, "
            "response quality, safety). Judges are calibrated against human labels using Cohen's kappa — "
            "Shopify accepts automated labels only when kappa ≥ 0.80 (strong inter-rater agreement). "
            "Low-scoring conversations (below threshold) become 'hard negatives'."
        ),
        "code_pattern": "scores = [judge.score(conv, rubric) for judge in judges]\nif all_agree(scores, threshold=0.8): accept_label()",
    },
    {
        "step": 3,
        "icon": "🔍",
        "title": "Critique: Frontier Reasoning Model Analyzes Failures",
        "detail": (
            "A frontier reasoning model (e.g., o3, Claude Sonnet) critiques each low-scoring conversation. "
            "It identifies WHY the output failed (missed context cue, wrong inference, impossible request) "
            "and writes a specific repair instruction. Multiple critiques are combined by an arbiter into "
            "a single actionable repair directive."
        ),
        "code_pattern": "critique = reasoning_model.critique(failed_conv, rubric)\nrepair_instruction = arbiter.combine(critiques)",
    },
    {
        "step": 4,
        "icon": "🔧",
        "title": "Repair: Replay Conversation with Fix Applied",
        "detail": (
            "The repair instruction is applied and the conversation is replayed. If the repaired version "
            "passes the LLM evaluator, it enters the training set as a positive example. "
            "Cases that still fail after repair go to human annotators. "
            "Shopify runs this daily for the GraphQL agent, accumulating repaired trajectories "
            "in a growing training set."
        ),
        "code_pattern": "repaired = apply_repair(conv, repair_instruction)\nif evaluator.score(repaired) >= threshold: add_to_training_set(repaired)",
    },
    {
        "step": 5,
        "icon": "🏋️",
        "title": "Train: SFT + GRPO on Accumulated Trajectories",
        "detail": (
            "The accumulated training set (repaired trajectories + successful production examples) "
            "is used to fine-tune a smaller model via: "
            "(1) Supervised Fine-Tuning (SFT) — model learns to imitate successful trajectories; "
            "(2) GRPO (Group Relative Policy Optimization) — model generates multiple responses, "
            "judges score them, and training increases probability of higher-scoring responses. "
            "Daily retraining on new + accumulated examples prevents catastrophic forgetting."
        ),
        "code_pattern": "model = sft(base_model, training_set)\nmodel = grpo(model, reward_fn=evaluator.score)",
    },
]


# ---------------------------------------------------------------------------
# Specialization checklist
# ---------------------------------------------------------------------------

SPECIALIZATION_CONDITIONS: list[dict] = [
    {
        "condition": "High volume",
        "icon": "📊",
        "detail": "The task runs at scale — thousands+ requests per day. Specialization's cost savings only matter when multiplied across millions of requests.",
        "shopify_example": "Buyer-profile generation: 2M/day → 72M/day. At this scale, $0.027/1K vs $37/1K = $26M annual savings.",
        "ready": True,
    },
    {
        "condition": "Bounded behavior",
        "icon": "🎯",
        "detail": "The task has a well-defined input/output contract. The frontier model's flexibility is no longer needed — the task distribution has stabilized.",
        "shopify_example": "GraphQL agent: merchant asks → agent writes query → executes → returns plain-language answer. Fixed contract.",
        "ready": True,
    },
    {
        "condition": "Measurable quality",
        "icon": "📏",
        "detail": "You can score outputs automatically. Without reliable automated evals, you can't build the flywheel — human labeling doesn't scale to millions of conversations.",
        "shopify_example": "Rubric: completeness, execution, response_quality, safety. LLM judges calibrated with Cohen's kappa ≥ 0.80.",
        "ready": True,
    },
    {
        "condition": "Proprietary production data",
        "icon": "🔒",
        "detail": "You have real failure trajectories that no synthetic dataset can replicate. Production 'hard negatives' capture your specific failure modes.",
        "shopify_example": "Millions of real merchant-customer interactions, including edge cases like impossible date-range queries that never appear in open datasets.",
        "ready": True,
    },
    {
        "condition": "Still in exploration phase?",
        "icon": "⚠️",
        "detail": "If the product is still changing (new tools, new user patterns, evolving requirements), stay on the frontier model. Training a specialist now locks in a distribution that may be obsolete in 2 weeks.",
        "shopify_example": "Shopify ran Sidekick on frontier models for months before the task distribution was stable enough to specialize.",
        "ready": False,
    },
]


def format_specialization_table() -> str:
    rows = [
        "| # | Condition | Status | Shopify Example |",
        "|---|-----------|--------|-----------------|",
    ]
    for i, c in enumerate(SPECIALIZATION_CONDITIONS, 1):
        status = "✅ Ready" if c["ready"] else "⛔ Not yet"
        rows.append(f"| {i} | {c['icon']} **{c['condition']}** | {status} | _{c['shopify_example'][:80]}..._ |")
    return "\n".join(rows)
