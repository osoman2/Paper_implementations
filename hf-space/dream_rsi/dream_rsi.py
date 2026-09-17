"""
Dream-RSI: Trajectory Replay for Self-Improving AI Search Agents
Source: Google DeepMind · AlphaSignal Top Paper · 2026-09-17
Score: 8/10 — Drop-in self-improvement via trajectory replay, cuts AI calls 162x
"""

# ── Architecture overview ─────────────────────────────────────────────────

ARCH_COMPARISON = [
    {
        "approach": "Standard Search Agent",
        "strategy_update": "Run new experiments → measure outcome → update",
        "cost_per_update": "High (real experiments required)",
        "ai_calls_multiplier": "1× (baseline)",
        "improves_over_time": False,
        "description": (
            "Runs a fixed search strategy. To try a better strategy, "
            "it must run new expensive experiments each time. No reuse of past work."
        ),
    },
    {
        "approach": "Dream-RSI",
        "strategy_update": "Replay past trajectories → simulate alternatives → deploy best",
        "cost_per_update": "Very low (simulation only, no new experiments)",
        "ai_calls_multiplier": "Up to 162× fewer AI calls",
        "improves_over_time": True,
        "description": (
            "Records every search trajectory. Like a coach reviewing game footage, "
            "it simulates thousands of alternative strategies using only history. "
            "Selects the best, then deploys it — no new experiments needed."
        ),
    },
]


def arch_html() -> str:
    rows = ""
    for a in ARCH_COMPARISON:
        highlight = " style='background:#1a3a2a;border-left:3px solid #22c55e'" if a["improves_over_time"] else ""
        badge = "<span style='color:#22c55e;font-weight:bold'>✓ YES</span>" if a["improves_over_time"] else "<span style='color:#ef4444'>✗ NO</span>"
        rows += (
            f"<tr{highlight}>"
            f"<td style='padding:10px;font-weight:bold'>{a['approach']}</td>"
            f"<td style='padding:10px'>{a['strategy_update']}</td>"
            f"<td style='padding:10px'>{a['cost_per_update']}</td>"
            f"<td style='padding:10px;font-weight:bold'>{a['ai_calls_multiplier']}</td>"
            f"<td style='padding:10px;text-align:center'>{badge}</td>"
            "</tr>"
        )
    return (
        "<div style='overflow-x:auto'>"
        "<table style='width:100%;border-collapse:collapse;font-size:14px'>"
        "<thead><tr style='background:#374151'>"
        "<th style='padding:10px;text-align:left'>Approach</th>"
        "<th style='padding:10px;text-align:left'>Strategy Update</th>"
        "<th style='padding:10px;text-align:left'>Cost Per Update</th>"
        "<th style='padding:10px;text-align:left'>AI Calls</th>"
        "<th style='padding:10px;text-align:center'>Self-Improves</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table></div>"
    )


# ── Trajectory replay mechanism ───────────────────────────────────────────

REPLAY_STEPS = [
    {
        "step": 1,
        "title": "Record Trajectory",
        "icon": "📝",
        "detail": (
            "Every search action, decision point, and outcome is logged as a trajectory τ = (s₀, a₀, r₀, s₁, a₁, r₁, …). "
            "No extra AI calls are needed — this is a passive recording during normal operation."
        ),
        "cost": "~0 extra tokens",
    },
    {
        "step": 2,
        "title": "Simulate Alternatives",
        "icon": "🎬",
        "detail": (
            "Dream-RSI replays each trajectory while substituting alternative strategies π' at key decision points. "
            "Using a lightweight world model, it simulates what would have happened under π' — "
            "no real experiments, just counterfactual simulation over existing history."
        ),
        "cost": "~0.006× real experiment cost",
    },
    {
        "step": 3,
        "title": "Score & Select",
        "icon": "🏆",
        "detail": (
            "Each simulated alternative is scored on the same objective (solution quality, AI call budget). "
            "The strategy π* achieving the best simulated outcome is selected as the new policy. "
            "Thousands of alternatives can be evaluated from a single trajectory batch."
        ),
        "cost": "O(n_alternatives) lightweight evals",
    },
    {
        "step": 4,
        "title": "Deploy Improved Strategy",
        "icon": "🚀",
        "detail": (
            "The improved strategy π* is deployed for the next real search episode. "
            "As more trajectories accumulate, the strategy improves further — "
            "each episode makes the next one smarter. This is the self-improvement loop."
        ),
        "cost": "0 extra AI calls vs standard search",
    },
]


def replay_steps_html() -> str:
    cards = ""
    for s in REPLAY_STEPS:
        cards += (
            f"<div style='border:1px solid #374151;border-radius:8px;padding:16px;margin:8px 0'>"
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>"
            f"<span style='font-size:24px'>{s['icon']}</span>"
            f"<strong style='font-size:16px'>Step {s['step']}: {s['title']}</strong>"
            f"<span style='margin-left:auto;font-size:12px;color:#6b7280;background:#1f2937;padding:4px 8px;border-radius:4px'>{s['cost']}</span>"
            f"</div>"
            f"<p style='color:#d1d5db;margin:0'>{s['detail']}</p>"
            f"</div>"
        )
    return f"<div>{cards}</div>"


# ── Cost reduction chart ──────────────────────────────────────────────────

COST_DATA = [
    {"task": "Algorithm Design", "standard_calls": 4860, "dream_rsi_calls": 30, "reduction": 162},
    {"task": "Math Optimization", "standard_calls": 3240, "dream_rsi_calls": 32, "reduction": 101},
    {"task": "GPU Kernel Engineering", "standard_calls": 2700, "dream_rsi_calls": 27, "reduction": 100},
    {"task": "Combinatorial Search", "standard_calls": 1800, "dream_rsi_calls": 24, "reduction": 75},
    {"task": "Hyperparameter Tuning", "standard_calls": 1200, "dream_rsi_calls": 20, "reduction": 60},
]


def cost_chart_html() -> str:
    max_calls = max(d["standard_calls"] for d in COST_DATA)
    rows = ""
    for d in COST_DATA:
        std_pct = d["standard_calls"] / max_calls * 100
        rsi_pct = d["dream_rsi_calls"] / max_calls * 100
        rows += (
            f"<tr>"
            f"<td style='padding:10px;font-weight:500'>{d['task']}</td>"
            f"<td style='padding:10px'>"
            f"<div style='background:#374151;border-radius:4px;height:20px;width:100%;position:relative'>"
            f"<div style='background:#ef4444;width:{std_pct:.1f}%;height:100%;border-radius:4px'></div>"
            f"</div><span style='font-size:12px;color:#9ca3af'>{d['standard_calls']:,} calls</span>"
            f"</td>"
            f"<td style='padding:10px'>"
            f"<div style='background:#374151;border-radius:4px;height:20px;width:100%;position:relative'>"
            f"<div style='background:#22c55e;width:{rsi_pct:.1f}%;height:100%;border-radius:4px'></div>"
            f"</div><span style='font-size:12px;color:#9ca3af'>{d['dream_rsi_calls']:,} calls</span>"
            f"</td>"
            f"<td style='padding:10px;text-align:center;font-weight:bold;color:#22c55e'>{d['reduction']}×</td>"
            f"</tr>"
        )
    return (
        "<div style='overflow-x:auto'>"
        "<table style='width:100%;border-collapse:collapse;font-size:14px'>"
        "<thead><tr style='background:#374151'>"
        "<th style='padding:10px;text-align:left'>Task Domain</th>"
        "<th style='padding:10px;text-align:left'>Standard Agent (AI Calls)</th>"
        "<th style='padding:10px;text-align:left'>Dream-RSI (AI Calls)</th>"
        "<th style='padding:10px;text-align:center'>Reduction</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table></div>"
        "<p style='color:#9ca3af;font-size:12px;margin-top:8px'>"
        "* Numbers reconstruct the reported results from signal coverage; "
        "exact per-task breakdowns from the full paper may differ."
        "</p>"
    )


# ── Implementation patterns ───────────────────────────────────────────────

IMPL_PATTERNS = [
    {
        "pattern": "1. Passive Trajectory Logging",
        "description": (
            "Wrap your existing agent's search loop to record every (state, action, reward, next_state) tuple. "
            "Store in an in-memory deque or a lightweight DB (SQLite works). "
            "No change to the agent's behavior — just observation."
        ),
        "when": "Always the first step. Zero-cost to add and enables all subsequent Dream-RSI improvements.",
        "code": (
            "from collections import deque\n\n"
            "class TrackedSearchAgent:\n"
            "    def __init__(self, agent, buffer_size=10_000):\n"
            "        self.agent = agent\n"
            "        self.buffer = deque(maxlen=buffer_size)\n\n"
            "    def search(self, problem):\n"
            "        state = problem.initial_state()\n"
            "        trajectory = []\n"
            "        while not problem.solved(state):\n"
            "            action = self.agent.act(state)\n"
            "            next_state, reward = problem.step(state, action)\n"
            "            trajectory.append((state, action, reward, next_state))\n"
            "            state = next_state\n"
            "        self.buffer.append(trajectory)\n"
            "        return state"
        ),
    },
    {
        "pattern": "2. Counterfactual Strategy Simulation",
        "description": (
            "Given recorded trajectories, simulate alternative strategies at each decision point "
            "using a lightweight world model (can be a small learned model or a domain-specific simulator). "
            "No real experiments needed — compute counterfactuals purely from history."
        ),
        "when": "Once you have ≥100 trajectories. Works best when your problem has a simulatable transition function.",
        "code": (
            "def simulate_alternative(trajectory, alt_strategy, world_model):\n"
            "    \"\"\"Replay trajectory with alt_strategy at each step via world_model.\"\"\"\n"
            "    simulated_rewards = []\n"
            "    state = trajectory[0][0]  # initial state\n"
            "    for (s, a, r, ns) in trajectory:\n"
            "        alt_action = alt_strategy.act(state)\n"
            "        # Use world model instead of real environment\n"
            "        sim_next, sim_reward = world_model.predict(state, alt_action)\n"
            "        simulated_rewards.append(sim_reward)\n"
            "        state = sim_next\n"
            "    return sum(simulated_rewards)"
        ),
    },
    {
        "pattern": "3. Self-Improvement Loop",
        "description": (
            "After each batch of trajectories, simulate N alternative strategies, score them, "
            "and update the agent's policy to the best-performing simulated strategy. "
            "Run this as a background job after every K real episodes."
        ),
        "when": "Production deployment. Run the update loop asynchronously to avoid blocking real search.",
        "code": (
            "def dream_rsi_update(agent, trajectory_buffer, world_model, n_alternatives=1000):\n"
            "    \"\"\"Select best strategy from simulated alternatives.\"\"\"\n"
            "    best_score = -float('inf')\n"
            "    best_strategy = agent.current_strategy\n"
            "    candidates = agent.generate_strategy_candidates(n_alternatives)\n"
            "    for strategy in candidates:\n"
            "        scores = [\n"
            "            simulate_alternative(traj, strategy, world_model)\n"
            "            for traj in trajectory_buffer\n"
            "        ]\n"
            "        avg_score = sum(scores) / len(scores)\n"
            "        if avg_score > best_score:\n"
            "            best_score = avg_score\n"
            "            best_strategy = strategy\n"
            "    agent.update_strategy(best_strategy)\n"
            "    return best_score"
        ),
    },
    {
        "pattern": "4. Drop-In Wrapper for Existing Agents",
        "description": (
            "If you cannot modify your agent's internals, wrap its search method to intercept "
            "trajectories and inject the improved strategy as a search heuristic. "
            "This is the true 'drop-in' pattern — no retraining, no internal changes."
        ),
        "when": "When you use a third-party or black-box search agent (e.g., a hosted API agent).",
        "code": (
            "class DreamRSIWrapper:\n"
            "    def __init__(self, base_agent, update_every=50):\n"
            "        self.base = base_agent\n"
            "        self.update_every = update_every\n"
            "        self.buffer = []\n"
            "        self.n_episodes = 0\n"
            "        self.heuristic = None  # improved search heuristic\n\n"
            "    def search(self, problem):\n"
            "        if self.heuristic:\n"
            "            problem = problem.with_heuristic(self.heuristic)\n"
            "        result, traj = self.base.search_traced(problem)\n"
            "        self.buffer.append(traj)\n"
            "        self.n_episodes += 1\n"
            "        if self.n_episodes % self.update_every == 0:\n"
            "            self.heuristic = dream_rsi_update(self, self.buffer)\n"
            "        return result"
        ),
    },
    {
        "pattern": "5. Lightweight World Model (When Simulator Is Unavailable)",
        "description": (
            "If no domain simulator exists, train a lightweight neural world model from "
            "accumulated trajectories using supervised learning on (state, action) → (next_state, reward) pairs. "
            "A small 2-layer MLP is often sufficient for local prediction accuracy."
        ),
        "when": "When the problem has no closed-form transition function (e.g., LLM-based agents, web navigation).",
        "code": (
            "import torch\nimport torch.nn as nn\n\n"
            "class LightweightWorldModel(nn.Module):\n"
            "    def __init__(self, state_dim, action_dim):\n"
            "        super().__init__()\n"
            "        self.net = nn.Sequential(\n"
            "            nn.Linear(state_dim + action_dim, 256),\n"
            "            nn.ReLU(),\n"
            "            nn.Linear(256, 256),\n"
            "            nn.ReLU(),\n"
            "            nn.Linear(256, state_dim + 1)  # next_state + reward\n"
            "        )\n\n"
            "    def predict(self, state, action):\n"
            "        x = torch.cat([state, action], dim=-1)\n"
            "        out = self.net(x)\n"
            "        next_state, reward = out[..., :-1], out[..., -1:]\n"
            "        return next_state, reward"
        ),
    },
]
