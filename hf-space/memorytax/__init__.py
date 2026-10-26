"""
MemoryTax: when is an agent memory layer worth paying for? — POC (Tab 53)
Source: AlphaSignal Sunday Deep Dive (Ben Dickson) · 2026-10-25 · news@alphasignal.ai
Score: 8/10

MemoryTax (Shanghai Jiao Tong University) compares seven ways an agent can access a long
conversation (LoCoMo, one conversation, 152 questions): full context, BM25 retrieval, Mem0,
Zep and agent-directed search with coding harnesses. Figures below are the ones reported in
the newsletter summary; the demo adds a break-even calculator and a decision helper.
"""

MODEL_ROWS = [
    ["GPT-4o-mini", "Mem0 (best)", "Memory extraction helps the weakest model"],
    ["GPT-6 Luna", "Full context (best)", "Raw history beats the memory layer"],
    ["Claude Opus 5.5", "Full context or Mem0 (both 95%)", "No accuracy gain from memory"],
]

CROSS_ROWS = [
    ["GPT-4o-mini", "GPT-4o-mini", 43],
    ["GPT-4o-mini", "Claude Opus 5.5", 45],
    ["Claude Opus 5.5", "Claude Opus 5.5", 87],
]

HARNESS_ROWS = [
    ["Pi (agent-directed search)", 99, 6.95],
    ["Full context", 95, 17.84],
    ["Codex", 98, 25.41],
    ["Claude Code", 97, 28.41],
]

CARDS = [
    {
        "name": "What was tested",
        "body": (
            "Seven access methods on one LoCoMo conversation (152 questions: recall, cross-message "
            "links, event order) with GPT-4o-mini, GPT-6 Luna and Claude Opus 5.5. Answers were judged "
            "by GPT-4o-mini. Methods: full context, BM25 retrieval, Mem0 (extracted facts), Zep "
            "(knowledge graph), and agent-directed search where Pi, Codex or Claude Code read the raw "
            "history as a JSON file."
        ),
    },
    {
        "name": "Finding 1: stronger models change the economics",
        "body": (
            "Mem0 won with GPT-4o-mini, full context won with GPT-6 Luna, and Opus 5.5 reached 95% "
            "with either. With GPT-4o-mini, full context for all 152 questions cost $0.43 while "
            "building the Mem0 store alone cost $0.56; Mem0 breaks even after 372-596 questions "
            "depending on the model."
        ),
    },
    {
        "name": "Finding 2: the writer sets the ceiling",
        "body": (
            "A memory store is limited by what the model that built it chose to keep. A Zep graph "
            "built by GPT-4o-mini scored 43% when read by GPT-4o-mini and 45% when read by Opus 5.5; "
            "built and read by Opus 5.5 it scored 87%. Keep the original records available."
        ),
    },
    {
        "name": "Finding 3: model-directed search lowers cost",
        "body": (
            "Opus 5.5 with the minimal Pi harness answered 99% of questions for $6.95 against 95% for "
            "full context at $17.84 (about 10k vs 28.9k input tokens per question). Codex ($25.41, "
            "98%) and Claude Code ($28.41, 97%) cost more on the same model."
        ),
    },
    {
        "name": "Caveats",
        "body": (
            "One conversation, one run per configuration, a GPT-4o-mini judge, and the whole history "
            "fit in every context window. Treat it as a test framework, not a verdict. Numbers here "
            "come from a newsletter summary of the study."
        ),
    },
]

IMPL_PATTERNS = [
    {
        "name": "Break-even before building a memory store",
        "description": "Compare the one-off build cost against per-question savings.",
        "code": '''def break_even(build_cost, full_ctx_per_q, memory_per_q):
    saving = full_ctx_per_q - memory_per_q
    return float("inf") if saving <= 0 else build_cost / saving
''',
    },
    {
        "name": "Keep raw records next to extracted facts",
        "description": "Fall back to the original history when the extracted store lacks the detail.",
        "code": '''def answer(question, facts, raw_history, llm):
    hit = facts.search(question)
    if hit and llm.confident(question, hit):
        return llm.answer(question, hit)
    return llm.answer(question, raw_history.grep(question))   # model-directed search
''',
    },
]


def _num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def break_even(build_cost, full_per_q, memory_per_q, questions):
    """Markdown verdict on whether a memory store pays for itself over `questions` questions."""
    build, full_q, mem_q, n = _num(build_cost), _num(full_per_q), _num(memory_per_q), _num(questions)
    if build <= 0 and full_q <= 0 and mem_q <= 0:
        return "Enter the build cost and the per-question costs to see the break-even point."
    saving = full_q - mem_q
    if saving <= 0:
        return ("The memory store is not cheaper per question than full context, "
                "so it never pays back its build cost.")
    be = build / saving
    full_total, mem_total = full_q * n, build + mem_q * n
    verdict = "pays back" if n >= be else "does **not** pay back"
    return (
        f"Break-even after **{be:,.0f} questions**.\n\n"
        f"| Option | Total cost for {n:,.0f} questions |\n|---|---|\n"
        f"| Full context | ${full_total:,.2f} |\n| Memory store (build + queries) | ${mem_total:,.2f} |\n\n"
        f"At this volume the memory layer {verdict} its build cost. "
        "Accuracy is not modelled: check it on your own data."
    )


def recommend(history_fits, questions, strong_model, reuse_stable):
    """Markdown recommendation following the four checks in the article."""
    n = _num(questions)
    steps = []
    if history_fits and n < 300:
        steps.append("**Start with full context.** The history fits and query volume is low.")
    elif history_fits:
        steps.append("**Measure full context first**, then compare against a memory store at your volume.")
    else:
        steps.append("**Add retrieval.** The history does not fit: try BM25 and model-directed file search first.")
    if strong_model:
        steps.append("A strong model reads raw history well; prefer agent-directed search over extraction.")
    else:
        steps.append("A weaker model can gain from structured memory (Mem0 won with GPT-4o-mini).")
    if reuse_stable and n >= 300:
        steps.append("Stable, repeatedly queried data can justify precomputed memory; let the strongest model write it.")
    steps.append("Keep the original records and benchmark accuracy, tokens and total cost on real tasks.")
    return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
