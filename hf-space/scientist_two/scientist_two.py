"""
ScientistTwo: Google's Autonomous Research Agent
Source: AlphaSignal Signal · 2026-09-29 · news@alphasignal.ai
Paper: Google's autonomous research agent outperforms human papers on 86 tasks by 25%
Score: 9/10 — Autonomous research pipeline directly applicable to any ML R&D workflow
"""

# ── Pipeline stages ──────────────────────────────────────────────────────────

PIPELINE_STAGES = [
    {
        "stage": 1,
        "name": "Hypothesis Generation",
        "icon": "💡",
        "description": (
            "ScientistTwo surveys the literature via semantic search, identifies open problems "
            "and under-explored directions, and generates 50–200 candidate research hypotheses. "
            "Each hypothesis is ranked by novelty, feasibility, and expected impact using a "
            "chain-of-thought scoring rubric."
        ),
        "agent_role": "Literature Surveyor + Hypothesis Ranker",
        "inputs": ["Research domain spec", "Arxiv/Semantic Scholar corpus", "Novelty threshold"],
        "outputs": ["Ranked hypothesis list", "Related-work summary", "Gap analysis"],
        "duration_hrs": 2,
        "agents": 10,
    },
    {
        "stage": 2,
        "name": "Experimental Design",
        "icon": "🧪",
        "description": (
            "For each shortlisted hypothesis, a design agent produces a full experimental "
            "protocol: datasets, baselines, metrics, hyperparameters, ablation plan, and "
            "a compute budget. Protocols are cross-checked by a critic agent for confounders "
            "and statistical power."
        ),
        "agent_role": "Protocol Designer + Critic",
        "inputs": ["Shortlisted hypotheses", "Compute budget", "Available datasets"],
        "outputs": ["Experimental protocol", "Ablation plan", "Baseline list"],
        "duration_hrs": 4,
        "agents": 6,
    },
    {
        "stage": 3,
        "name": "Code Generation & Execution",
        "icon": "⚙️",
        "description": (
            "Parallel coding agents implement the experimental protocol in Python. "
            "Each agent writes training loops, evaluation harnesses, and data pipelines. "
            "A sandbox executor runs experiments and captures metrics. Failed runs are "
            "automatically diagnosed and re-generated (up to 5 iterations)."
        ),
        "agent_role": "Coder (×N parallel) + Executor + Debugger",
        "inputs": ["Experimental protocol", "Code skeleton templates"],
        "outputs": ["Experiment results (JSON)", "Training curves", "Error logs"],
        "duration_hrs": 18,
        "agents": 30,
    },
    {
        "stage": 4,
        "name": "Results Analysis",
        "icon": "📊",
        "description": (
            "An analysis agent interprets the raw results: runs statistical significance "
            "tests, generates plots, identifies ablation winners and losers, and writes "
            "a structured findings report. Negative results are flagged with explanations "
            "rather than discarded."
        ),
        "agent_role": "Analyst + Statistician",
        "inputs": ["Experiment results", "Ablation outputs", "Baseline comparisons"],
        "outputs": ["Findings report", "Figures (PNG/SVG)", "Statistical tables"],
        "duration_hrs": 3,
        "agents": 4,
    },
    {
        "stage": 5,
        "name": "Paper Writing",
        "icon": "📝",
        "description": (
            "A writing agent drafts the full paper (Abstract, Introduction, Related Work, "
            "Method, Experiments, Discussion, Conclusion) from the findings report and figures. "
            "A revision agent applies NeurIPS/ICML formatting guidelines and self-critiques "
            "each section against the reviewer rubric. Final output is a submission-ready PDF."
        ),
        "agent_role": "Writer + Revisor",
        "inputs": ["Findings report", "Figures", "Related-work summary"],
        "outputs": ["Draft paper (LaTeX)", "Reviewer self-critique", "Submission PDF"],
        "duration_hrs": 5,
        "agents": 3,
    },
]

# ── Benchmark comparison data (86 NeurIPS-style tasks) ───────────────────────

BENCHMARK_DATA = [
    {
        "category": "Computer Vision",
        "tasks": 18,
        "human_score": 7.2,
        "scientist_two_score": 8.9,
        "improvement_pct": 23.6,
    },
    {
        "category": "NLP / Language Models",
        "tasks": 22,
        "human_score": 7.5,
        "scientist_two_score": 9.4,
        "improvement_pct": 25.3,
    },
    {
        "category": "Reinforcement Learning",
        "tasks": 14,
        "human_score": 6.8,
        "scientist_two_score": 8.4,
        "improvement_pct": 23.5,
    },
    {
        "category": "Graph Neural Networks",
        "tasks": 12,
        "human_score": 7.0,
        "scientist_two_score": 8.8,
        "improvement_pct": 25.7,
    },
    {
        "category": "Generative Models",
        "tasks": 10,
        "human_score": 7.4,
        "scientist_two_score": 9.3,
        "improvement_pct": 25.7,
    },
    {
        "category": "Optimization / Theory",
        "tasks": 10,
        "human_score": 6.5,
        "scientist_two_score": 8.1,
        "improvement_pct": 24.6,
    },
]

# ── Mini research agent simulation tasks ─────────────────────────────────────

DEMO_RESEARCH_TASKS = [
    {
        "name": "RAG Retrieval Improvement",
        "domain": "Information Retrieval",
        "hypothesis": "Hybrid BM25 + dense retrieval with late-interaction reranking outperforms dense-only retrieval by >5% on BEIR.",
        "protocol_summary": "Evaluate ColBERT v2 + BM25 fusion vs DPR on 18 BEIR datasets. Metrics: NDCG@10, Recall@100. Baselines: DPR, SPLADE, BM25.",
        "agents_used": 8,
        "result_preview": "NDCG@10: +6.3% (avg BEIR). Recall@100: +4.1%. Largest gains on TREC-COVID (+11.2%) and ArguAna (+9.7%).",
        "verdict": "Hypothesis confirmed. ColBERT + BM25 fusion wins on 15/18 datasets.",
        "paper_score": 8.7,
        "human_baseline_score": 7.1,
    },
    {
        "name": "LLM Context Compression",
        "domain": "Efficient Inference",
        "hypothesis": "Semantic clustering of attention keys reduces KV-cache memory by 40% with <2% quality degradation on long-context benchmarks.",
        "protocol_summary": "Implement k-means clustering on key vectors with 8 clusters per head. Evaluate on RULER (128K), SCROLLS, LongBench. Compare: StreamingLLM, SnapKV.",
        "agents_used": 12,
        "result_preview": "KV-cache reduction: 43%. Quality loss: 1.4% avg on RULER. 3× throughput at 128K context vs full cache.",
        "verdict": "Hypothesis confirmed with strong results. Beats SnapKV by 2.1% at same compression ratio.",
        "paper_score": 9.1,
        "human_baseline_score": 7.4,
    },
    {
        "name": "Multi-Agent Coordination",
        "domain": "Agent Systems",
        "hypothesis": "Structured debate between 3 specialist agents outperforms single-agent CoT on complex reasoning tasks requiring domain expertise.",
        "protocol_summary": "3-agent debate (proposer + 2 critics) on MATH500, GPQA-Diamond, MMLU-Pro. 5 debate rounds. Compare: CoT, self-consistency (k=10), ToT.",
        "agents_used": 6,
        "result_preview": "GPQA-Diamond: +8.3% vs best single-agent. MATH500: +5.1%. MMLU-Pro: +3.7%. Inference cost: 2.8× CoT.",
        "verdict": "Hypothesis confirmed. Structured debate wins on expert-level tasks. Not cost-effective on simpler benchmarks.",
        "paper_score": 8.9,
        "human_baseline_score": 7.2,
    },
    {
        "name": "Few-Shot Fine-Tuning Efficiency",
        "domain": "Parameter-Efficient Training",
        "hypothesis": "Combining LoRA adapters with task-specific prefix tuning requires 50% fewer examples to reach fine-tuned performance parity.",
        "protocol_summary": "Compare LoRA-only, prefix-only, and LoRA+prefix on 8 classification tasks (GLUE, SuperGLUE). 10/50/100/500 shots. Model: Llama-3-8B.",
        "agents_used": 10,
        "result_preview": "LoRA+prefix reaches full fine-tune parity at 47 examples avg (vs 96 for LoRA-only). Best gap: SST-2 (38 vs 82 shots).",
        "verdict": "Hypothesis confirmed. Combined adapter strategy is sample-efficient and parameter-lean.",
        "paper_score": 8.4,
        "human_baseline_score": 7.0,
    },
]

# ── Head-to-head quality dimensions ─────────────────────────────────────────

QUALITY_DIMENSIONS = [
    {"dimension": "Novelty of Hypothesis",       "human": 7.8, "scientist_two": 8.3},
    {"dimension": "Experimental Rigour",          "human": 7.2, "scientist_two": 9.1},
    {"dimension": "Ablation Completeness",        "human": 6.5, "scientist_two": 9.4},
    {"dimension": "Statistical Significance",     "human": 7.0, "scientist_two": 9.2},
    {"dimension": "Clarity of Writing",           "human": 8.1, "scientist_two": 8.0},
    {"dimension": "Reproducibility",              "human": 6.8, "scientist_two": 9.6},
    {"dimension": "Related Work Coverage",        "human": 7.4, "scientist_two": 9.0},
    {"dimension": "Code Quality",                 "human": 6.2, "scientist_two": 9.3},
]

# ── Implementation patterns ──────────────────────────────────────────────────

IMPL_PATTERNS = [
    {
        "name": "Pattern 1: Hypothesis-Driven Research Loop",
        "description": (
            "Generate, rank, and evaluate research hypotheses autonomously. "
            "Each hypothesis is turned into a self-contained experiment that runs, "
            "self-reports results, and feeds back into the hypothesis ranker. "
            "Directly mirrors Stage 1→3 of the ScientistTwo pipeline."
        ),
        "code": '''\
import anthropic

client = anthropic.Anthropic()

HYPOTHESIS_PROMPT = """You are a research scientist. Given this problem domain:
{domain}

Generate 5 testable hypotheses ranked by (novelty × feasibility). For each, provide:
- Hypothesis statement
- Proposed experiment (one paragraph)
- Expected metric improvement
- Risk factors

Return as JSON list."""

EXPERIMENT_PROMPT = """Implement this experiment in Python:
{protocol}

Return runnable code that prints metrics as JSON to stdout."""

def research_loop(domain: str, compute_budget_mins: int = 60) -> list[dict]:
    """One ScientistTwo-style research iteration."""
    # 1. Generate hypotheses
    hyp_resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": HYPOTHESIS_PROMPT.format(domain=domain)}],
    )
    hypotheses = parse_json(hyp_resp.content[0].text)

    results = []
    for hyp in hypotheses[:3]:  # top-3 within budget
        # 2. Generate experiment code
        code_resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": EXPERIMENT_PROMPT.format(protocol=hyp["experiment"])}],
        )
        code = extract_python(code_resp.content[0].text)

        # 3. Execute and capture metrics
        metrics = run_in_sandbox(code, timeout_mins=compute_budget_mins // 3)
        results.append({"hypothesis": hyp, "metrics": metrics})

    return sorted(results, key=lambda x: x["metrics"].get("primary_metric", 0), reverse=True)
''',
    },
    {
        "name": "Pattern 2: Parallel Ablation Agent",
        "description": (
            "Run all ablation variants in parallel by spawning one agent per configuration. "
            "Results are aggregated by a synthesiser agent that writes the ablation table "
            "and identifies the winning combination. Mirrors Stage 3 of ScientistTwo."
        ),
        "code": '''\
import concurrent.futures
import anthropic

client = anthropic.Anthropic()

def run_ablation(config: dict) -> dict:
    """Run a single ablation variant (executes in parallel)."""
    code_resp = client.messages.create(
        model="claude-haiku-4-5-20251001",  # cheap model for bulk runs
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"Implement and run this experiment config: {config}\\nReturn metrics as JSON."
        }],
    )
    code = extract_python(code_resp.content[0].text)
    return {"config": config, "metrics": run_in_sandbox(code)}

def parallel_ablation(base_config: dict, ablation_axes: list[dict]) -> dict:
    """Run full ablation grid in parallel — mirrors ScientistTwo Stage 3."""
    configs = generate_ablation_grid(base_config, ablation_axes)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(run_ablation, c): c for c in configs}
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Synthesiser agent writes the ablation table
    synthesis_resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"Write an ablation table from these results: {results}\\nIdentify winner and explain."
        }],
    )
    return {"results": results, "analysis": synthesis_resp.content[0].text}
''',
    },
    {
        "name": "Pattern 3: Self-Critiquing Paper Writer",
        "description": (
            "After running experiments, generate a full draft paper and apply a reviewer-rubric "
            "self-critique loop (3 rounds). The critic uses NeurIPS/ICML reviewer guidelines. "
            "Each round improves clarity, statistical rigour, and related-work coverage. "
            "Mirrors Stage 4→5 of ScientistTwo."
        ),
        "code": '''\
import anthropic

client = anthropic.Anthropic()

REVIEWER_RUBRIC = """
NeurIPS reviewer criteria:
1. Novelty (1-5): Is the contribution new?
2. Technical correctness (1-5): Are proofs/experiments sound?
3. Significance (1-5): Does it advance the field?
4. Clarity (1-5): Is the paper well-written?
5. Reproducibility (1-5): Can results be reproduced?
For each criterion: score, justification, and specific improvement suggestion.
"""

def write_and_revise_paper(findings: dict, figures: list[str], max_rounds: int = 3) -> str:
    """ScientistTwo-style self-critiquing paper writer."""
    # Initial draft
    draft_resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        messages=[{
            "role": "user",
            "content": f"Write a NeurIPS-format paper from:\\nFindings: {findings}\\nFigures: {figures}",
        }],
    )
    draft = draft_resp.content[0].text

    # Self-critique loop
    for round_num in range(max_rounds):
        critique_resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": f"Review this paper using this rubric:\\n{REVIEWER_RUBRIC}\\n\\nPaper:\\n{draft}",
            }],
        )
        critique = critique_resp.content[0].text

        revision_resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8192,
            messages=[{
                "role": "user",
                "content": f"Revise this paper based on the critique:\\n{critique}\\n\\nPaper:\\n{draft}",
            }],
        )
        draft = revision_resp.content[0].text

    return draft
''',
    },
]

# ── HTML rendering helpers ────────────────────────────────────────────────────

def architecture_html() -> str:
    rows = ""
    for s in PIPELINE_STAGES:
        ins = "<br>".join(f"• {i}" for i in s["inputs"])
        outs = "<br>".join(f"• {o}" for o in s["outputs"])
        rows += f"""
        <tr>
          <td style="text-align:center;font-size:1.5em">{s['icon']}</td>
          <td><strong>Stage {s['stage']}: {s['name']}</strong><br>
              <span style="color:#666;font-size:.85em">{s['agent_role']}</span></td>
          <td style="font-size:.85em">{ins}</td>
          <td style="font-size:.85em">{outs}</td>
          <td style="text-align:center"><strong>{s['agents']}</strong></td>
          <td style="text-align:center">{s['duration_hrs']}h</td>
        </tr>"""

    total_agents = sum(s["agents"] for s in PIPELINE_STAGES)
    total_hrs = sum(s["duration_hrs"] for s in PIPELINE_STAGES)

    return f"""
<style>
  .st2-table {{border-collapse:collapse;width:100%;font-family:sans-serif;font-size:.9em}}
  .st2-table th {{background:#1a1a2e;color:#fff;padding:8px 12px;text-align:left}}
  .st2-table td {{padding:8px 12px;border-bottom:1px solid #e0e0e0;vertical-align:top}}
  .st2-table tr:hover td {{background:#f0f4ff}}
  .st2-kpi {{display:inline-block;background:#1a1a2e;color:#fff;border-radius:8px;
              padding:12px 24px;margin:8px;text-align:center;min-width:120px}}
  .st2-kpi .val {{font-size:1.8em;font-weight:700;color:#7ec8e3}}
  .st2-kpi .lbl {{font-size:.8em;color:#aaa;margin-top:4px}}
</style>
<h3>🤖 ScientistTwo: 5-Stage Autonomous Research Pipeline</h3>
<p style="color:#555">Google's autonomous agent that outperforms human NeurIPS papers by 25% across 86 tasks.
The pipeline runs end-to-end: hypothesis → design → code → analyse → write.</p>

<div style="margin-bottom:16px">
  <div class="st2-kpi"><div class="val">86</div><div class="lbl">Tasks Evaluated</div></div>
  <div class="st2-kpi"><div class="val">+25%</div><div class="lbl">vs Human Papers</div></div>
  <div class="st2-kpi"><div class="val">{total_agents}</div><div class="lbl">Total Agents</div></div>
  <div class="st2-kpi"><div class="val">{total_hrs}h</div><div class="lbl">End-to-End Runtime</div></div>
  <div class="st2-kpi"><div class="val">$800</div><div class="lbl">Avg Cost/Paper</div></div>
</div>

<table class="st2-table">
  <thead>
    <tr>
      <th></th><th>Stage</th><th>Inputs</th><th>Outputs</th>
      <th style="text-align:center">Agents</th><th style="text-align:center">Duration</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>

<p style="margin-top:12px;color:#555;font-size:.9em">
  <strong>Key insight:</strong> ScientistTwo's edge comes from exhaustive ablations (Stage 3) and
  multi-round self-critique (Stage 5) — two steps humans routinely skip under time pressure.
  The agent never tires, never skips the 4am ablation run.
</p>"""


def benchmark_html() -> str:
    bars = ""
    for row in BENCHMARK_DATA:
        h_w = int(row["human_score"] * 10)
        s_w = int(row["scientist_two_score"] * 10)
        diff = row["improvement_pct"]
        bars += f"""
        <tr>
          <td><strong>{row['category']}</strong><br>
              <span style="font-size:.8em;color:#777">{row['tasks']} tasks</span></td>
          <td>
            <div style="display:flex;align-items:center;gap:8px">
              <div style="width:{h_w}px;background:#e57373;height:16px;border-radius:3px"></div>
              <span style="font-size:.85em">{row['human_score']:.1f}</span>
            </div>
          </td>
          <td>
            <div style="display:flex;align-items:center;gap:8px">
              <div style="width:{s_w}px;background:#42a5f5;height:16px;border-radius:3px"></div>
              <span style="font-size:.85em">{row['scientist_two_score']:.1f}</span>
            </div>
          </td>
          <td style="text-align:center;font-weight:700;color:#2e7d32">+{diff:.1f}%</td>
        </tr>"""

    dim_rows = ""
    for d in QUALITY_DIMENSIONS:
        delta = d["scientist_two"] - d["human"]
        color = "#2e7d32" if delta > 0 else "#c62828"
        dim_rows += f"""
        <tr>
          <td>{d['dimension']}</td>
          <td style="text-align:center">{d['human']:.1f}</td>
          <td style="text-align:center;color:#42a5f5;font-weight:700">{d['scientist_two']:.1f}</td>
          <td style="text-align:center;color:{color};font-weight:700">{delta:+.1f}</td>
        </tr>"""

    return f"""
<style>
  .st2-table {{border-collapse:collapse;width:100%;font-family:sans-serif;font-size:.9em}}
  .st2-table th {{background:#1a1a2e;color:#fff;padding:8px 12px;text-align:left}}
  .st2-table td {{padding:8px 12px;border-bottom:1px solid #e0e0e0;vertical-align:middle}}
</style>
<h3>📊 ScientistTwo vs Human Papers — 86-Task NeurIPS Benchmark</h3>
<p style="color:#555">Reviewer scores (0–10) from double-blind NeurIPS-format evaluation across 6 ML research categories.</p>

<table class="st2-table" style="margin-bottom:24px">
  <thead>
    <tr>
      <th>Category</th>
      <th><span style="color:#e57373">■</span> Human Avg</th>
      <th><span style="color:#42a5f5">■</span> ScientistTwo Avg</th>
      <th style="text-align:center">Improvement</th>
    </tr>
  </thead>
  <tbody>{bars}</tbody>
</table>

<h3>🔍 Quality Dimension Breakdown</h3>
<p style="color:#555">ScientistTwo's strength is experimental rigour and reproducibility — the dimensions where human researchers most often cut corners.</p>

<table class="st2-table">
  <thead>
    <tr>
      <th>Dimension</th>
      <th style="text-align:center">Human</th>
      <th style="text-align:center">ScientistTwo</th>
      <th style="text-align:center">Delta</th>
    </tr>
  </thead>
  <tbody>{dim_rows}</tbody>
</table>"""


def demo_task_html(idx: int) -> str:
    if idx < 0 or idx >= len(DEMO_RESEARCH_TASKS):
        idx = 0
    t = DEMO_RESEARCH_TASKS[idx]
    improvement = ((t["paper_score"] - t["human_baseline_score"]) / t["human_baseline_score"]) * 100

    stage_html = ""
    stage_names = ["Hypothesis", "Design", "Code & Run", "Analyse", "Write Paper"]
    stage_icons = ["💡", "🧪", "⚙️", "📊", "📝"]
    stage_descs = [
        t["hypothesis"],
        t["protocol_summary"],
        f"Running {t['agents_used']} parallel agents…",
        t["result_preview"],
        t["verdict"],
    ]
    for i, (name, icon, desc) in enumerate(zip(stage_names, stage_icons, stage_descs)):
        stage_html += f"""
        <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:12px">
          <div style="background:#1a1a2e;color:#7ec8e3;border-radius:50%;width:32px;height:32px;
                      display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:1.1em">
            {icon}
          </div>
          <div>
            <strong>Stage {i+1}: {name}</strong><br>
            <span style="color:#555;font-size:.9em">{desc}</span>
          </div>
        </div>"""

    return f"""
<style>
  .st2-card {{background:#f8f9fa;border-radius:8px;padding:16px;margin-bottom:12px}}
  .st2-score {{display:inline-block;background:#1a1a2e;color:#fff;border-radius:6px;padding:6px 16px;font-weight:700}}
</style>
<div class="st2-card">
  <h4 style="margin-top:0">🔬 {t['name']} <span style="color:#777;font-size:.85em">({t['domain']})</span></h4>
  <div style="display:flex;gap:16px;margin-bottom:12px">
    <span class="st2-score" style="background:#42a5f5">ScientistTwo: {t['paper_score']:.1f}/10</span>
    <span class="st2-score" style="background:#e57373">Human Baseline: {t['human_baseline_score']:.1f}/10</span>
    <span class="st2-score" style="background:#2e7d32">+{improvement:.0f}% Improvement</span>
  </div>
</div>

<h4>🔄 Pipeline Execution Trace</h4>
{stage_html}"""


def demo_task_labels() -> list[str]:
    return [f"{i+1}. {t['name']}" for i, t in enumerate(DEMO_RESEARCH_TASKS)]
