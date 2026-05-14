# CAD Score: Context-Aware Semantic Diversity

## Paper

| Field | Detail |
|-------|--------|
| **Paper** | CAD score: context-aware semantic diversity for scalable creativity evaluation of LLMs |
| **Authors** | Kim et al., Konkuk University, South Korea |
| **Journal** | PeerJ Computer Science, 2025 |

## Core Idea

LLMs tend to produce **low semantic diversity** even when asked for multiple different ideas — they generate slight variations on the same core concept. Existing metrics either miss this or fail to account for context:

- **Distinct-N**: counts unique n-grams. Misses semantic similarity (two texts can share 0 words yet mean the same thing).
- **Embedding spread**: mean pairwise cosine distance between outputs. Ignores whether variation is *meaningful* or just prompt paraphrasing.
- **Self-BLEU**: lexical overlap. Same blind spot as Distinct-N.

**CAD score** fixes this with a context penalty:

```
CAD = mean_pairwise_distance(responses) × (1 − mean_sim(responses, prompt))
```

If responses are diverse from each other but all closely echo the prompt → context penalty is low → CAD is low. You only get a high CAD when outputs are both *spread apart from each other* **and** *genuinely beyond* the prompt's scope.

## Relevance for AI Engineering

- **Prompt evaluation**: compare prompts objectively — which system prompt yields the most creative and diverse outputs?
- **Model selection**: rank models on creative tasks without human judges
- **Sampling strategy tuning**: evaluate effect of temperature, top-p, different sampling configs
- **Agentic systems**: measure if a brainstorming agent is actually exploring the solution space or looping on similar ideas
- **No LLM judge needed** — pure embedding math, runs in milliseconds

## How to Run

```bash
cd cad-score
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Without API key (manual mode only):
python src/app.py

# With live LLM generation:
ANTHROPIC_API_KEY=sk-... python src/app.py
```

Then open `http://localhost:7860`.

The **Manual tab** works without an API key — paste any set of responses to score them.

## What's Implemented

### `src/cad_score.py`

**`cad_score(prompt, responses)`**
Core metric. Returns:
- `cad` — the main score
- `raw_spread` — mean pairwise cosine distance (context-unaware)
- `context_penalty` — `1 − mean(sim(response_i, prompt))`
- `mean_prompt_similarity` — how much outputs echo the prompt
- `pairwise_distances` — per-pair breakdown
- `prompt_similarities` — per-response similarity to prompt

**Baseline metrics:**
- `distinct_n(responses, n)` — ratio of unique n-grams to total
- `embedding_spread(responses)` — cosine spread without context adjustment
- `self_bleu(responses, n)` — inverse lexical overlap

### `src/llm_client.py`
Two Anthropic API wrappers:
- `generate_responses(prompt, n)` — plain generation
- `generate_responses_high_temp(prompt, n)` — with creativity system prompt that explicitly instructs different angles per response

### `src/app.py`
Gradio demo with two tabs:
- **Live**: generate N responses via Claude API, then score immediately
- **Manual**: paste your own responses (no API key needed)

## Example Output

For prompt `"Write an opening line for a mystery novel."` with 5 responses:

| Metric | Value |
|--------|-------|
| CAD score | 0.18 |
| Raw spread | 0.29 |
| Context penalty | 0.62 |
| Distinct-2 | 0.91 |
| Self-BLEU-2 | 0.02 |

Compare with a low-diversity case (5 near-identical responses):

| Metric | Value |
|--------|-------|
| CAD score | 0.03 |
| Raw spread | 0.08 |
| Context penalty | 0.41 |

## Interpreting Scores

| CAD range | Interpretation |
|-----------|----------------|
| > 0.15 | Good diversity — outputs explore meaningfully different territory |
| 0.05 – 0.15 | Moderate — some variation but common thread remains |
| < 0.05 | Low — outputs are too similar or mostly echo the prompt |

## Architecture

```
prompt + responses
       │
       ▼
SentenceTransformer("all-MiniLM-L6-v2")
       │
       ├─► pairwise cosine distances (responses × responses)
       │         └─► raw_spread = mean(distances)
       │
       └─► prompt similarities (prompt × each response)
                 └─► context_penalty = 1 − mean(sims)
                           │
                           └─► CAD = raw_spread × context_penalty
```
