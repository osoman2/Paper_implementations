"""
CAD Score — Gradio demo.

Evaluates the semantic diversity of LLM outputs using the Context-Aware
Diversity (CAD) metric from Kim et al. 2025.

Run:
    ANTHROPIC_API_KEY=sk-... python src/app.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from cad_score import all_metrics
from llm_client import generate_responses, generate_responses_high_temp

HAS_API_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))

EXAMPLE_PROMPTS = [
    "Give me 3 different ways to explain recursion to a beginner.",
    "Write an opening line for a mystery novel.",
    "What are the benefits of exercise?",
    "Describe the colour blue without using the word blue.",
    "Propose a startup idea in the AI space.",
]

EXAMPLE_MANUAL = """\
The quick brown fox jumps over the lazy dog.
A swift auburn fox leaps across a sleepy hound.
The nimble red fox vaults over the resting dog.
A fast tawny fox bounds over the tired canine.
The agile fox hops above the dozing dog."""


def _fmt_pairwise(distances: list[float]) -> str:
    if not distances:
        return ""
    lines = [f"Pair {i+1}: {d:.4f}" for i, d in enumerate(distances)]
    return "\n".join(lines)


def evaluate_manual(prompt: str, raw_responses: str):
    responses = [r.strip() for r in raw_responses.strip().split("\n") if r.strip()]
    if len(responses) < 2:
        return "Need at least 2 responses (one per line).", "", "", ""
    try:
        m = all_metrics(prompt, responses)
    except Exception as e:
        return str(e), "", "", ""
    return _build_outputs(m, responses)


def evaluate_live(prompt: str, n_responses: int, use_creative_system: bool, progress=gr.Progress()):
    if not HAS_API_KEY:
        return (
            "Set ANTHROPIC_API_KEY environment variable to use live generation.",
            "", "", ""
        )
    if not prompt.strip():
        return "Enter a prompt first.", "", "", ""

    progress(0.1, desc="Generating responses…")
    try:
        fn = generate_responses_high_temp if use_creative_system else generate_responses
        responses = fn(prompt, n=n_responses)
    except Exception as e:
        return f"API error: {e}", "", "", ""

    progress(0.7, desc="Computing CAD score…")
    try:
        m = all_metrics(prompt, responses)
    except Exception as e:
        return str(e), "", "", ""

    progress(1.0)
    return _build_outputs(m, responses)


def _build_outputs(m: dict, responses: list[str]):
    # Scores summary
    scores_md = f"""
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **CAD score** | **{m['cad']}** | Higher = more context-aware diversity |
| Raw embedding spread | {m['raw_spread']} | Mean pairwise cosine distance |
| Context penalty | {m['context_penalty']} | 1 − mean(sim(response, prompt)) |
| Mean prompt similarity | {m['mean_prompt_similarity']} | How much responses echo the prompt |
| Distinct-2 | {m['distinct_2']} | Lexical diversity (unique bigrams ratio) |
| Embedding spread | {m['embedding_spread']} | Spread without context adjustment |
| Self-BLEU-2 | {m['self_bleu_2']} | Lexical overlap (lower = more diverse) |
""".strip()

    # Pairwise breakdown
    pair_md = "**Pairwise cosine distances between responses:**\n\n"
    pair_md += _fmt_pairwise(m["pairwise_distances"])

    # CAD explanation
    explain_md = f"""
**CAD = raw_spread × context_penalty**

`{m['raw_spread']} × {m['context_penalty']} = {m['cad']}`

**Why context penalty matters:**
If all responses are highly similar to the prompt (`mean_prompt_sim = {m['mean_prompt_similarity']}`),
they're probably just paraphrasing the question rather than exploring different ideas.
CAD discounts this type of surface variation by multiplying with `1 − prompt_sim`.

A high raw spread but low CAD means: the responses look different from each other,
but mostly because they're all restating the prompt in slightly different ways.
""".strip()

    # Responses display
    resp_text = "\n\n---\n\n".join(
        f"**Response {i+1}** (prompt_sim={m['prompt_similarities'][i]}):\n{r}"
        for i, r in enumerate(responses)
    )

    return scores_md, pair_md, explain_md, resp_text


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="CAD Score — LLM Diversity Evaluator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# CAD Score: Context-Aware Semantic Diversity\n"
        "**Paper: Kim et al. 2025 — PeerJ Computer Science**\n\n"
        "Measures how *meaningfully* diverse a set of LLM outputs are, relative to the prompt. "
        "Unlike raw embedding spread, CAD penalises responses that merely paraphrase the question "
        "rather than exploring genuinely different ideas."
    )

    with gr.Tabs():
        # --- Tab 1: Live LLM generation ---
        with gr.Tab("Live (Anthropic API)"):
            gr.Markdown(
                "Generate N responses to the same prompt and immediately score their diversity. "
                + ("⚠️ **ANTHROPIC_API_KEY not set** — set it to enable this tab." if not HAS_API_KEY else "✅ API key detected.")
            )
            with gr.Row():
                live_prompt = gr.Textbox(label="Prompt", lines=2, placeholder="Give me 5 startup ideas in the AI space.")
                with gr.Column():
                    n_resp = gr.Slider(2, 8, value=5, step=1, label="Number of responses")
                    creative_cb = gr.Checkbox(value=False, label="Use creativity system prompt")
            gr.Examples(EXAMPLE_PROMPTS, inputs=live_prompt)
            live_btn = gr.Button("Generate & Score", variant="primary")

            with gr.Row():
                live_scores = gr.Markdown(label="Scores")
                live_explain = gr.Markdown(label="CAD Explanation")
            with gr.Row():
                live_pairs = gr.Markdown(label="Pairwise Distances")
                live_resps = gr.Markdown(label="Generated Responses")

            live_btn.click(
                fn=evaluate_live,
                inputs=[live_prompt, n_resp, creative_cb],
                outputs=[live_scores, live_pairs, live_explain, live_resps],
            )

        # --- Tab 2: Manual paste ---
        with gr.Tab("Manual (paste your own)"):
            gr.Markdown("Paste your own responses (one per line) to score them — no API key needed.")
            with gr.Row():
                manual_prompt = gr.Textbox(label="Prompt (context for CAD)", lines=2,
                                           placeholder="The prompt that was given to the LLM")
                manual_responses = gr.Textbox(label="Responses (one per line)", lines=8,
                                              value=EXAMPLE_MANUAL)
            manual_btn = gr.Button("Compute CAD Score", variant="primary")

            with gr.Row():
                manual_scores = gr.Markdown(label="Scores")
                manual_explain = gr.Markdown(label="CAD Explanation")
            with gr.Row():
                manual_pairs = gr.Markdown(label="Pairwise Distances")
                manual_resps = gr.Markdown(label="Responses")

            manual_btn.click(
                fn=evaluate_manual,
                inputs=[manual_prompt, manual_responses],
                outputs=[manual_scores, manual_pairs, manual_explain, manual_resps],
            )

    gr.Markdown(
        "---\n"
        "**Interpreting scores:**  \n"
        "CAD > 0.15 → good diversity · CAD 0.05–0.15 → moderate · CAD < 0.05 → low (responses are too similar or echo the prompt)"
    )

if __name__ == "__main__":
    demo.launch()
