"""
Actor, Judge or Teacher? decision models in long-horizon agents (Tab 55)
Source: Hugging Face Daily Papers · 2026-11-13 · https://arxiv.org/abs/2611.17167
Score: 8/10

The paper reports that a decision model that acts poorly can still teach RL policies well when it
judges complete action trajectories (ALFWorld, WebShop). Only the abstract was used. Everything
below is a simulated illustration with synthetic numbers, not the paper's results.
"""
import random

FINDINGS = [
    {"name": "Three roles for one model",
     "body": "Actor: picks each action. Judge: scores whole trajectories. Teacher: its judgments "
             "train a smaller RL policy. The paper asks which roles a given decision model suits."},
    {"name": "Acting well is not teaching well",
     "body": "A model with a low success rate as actor can still be a useful teacher when it "
             "judges complete trajectories rather than single steps."},
    {"name": "Benchmarks",
     "body": "ALFWorld and WebShop. The paper reports improved success when RL policies are taught "
             "by trajectory-level judging. The numbers in this demo are synthetic."},
]

ROLE_ROWS = [
    ["Actor", "Per-step action", "Latency on the critical path", "Needs strong planning"],
    ["Judge", "Whole trajectory score", "Offline, batchable", "Needs outcome discrimination"],
    ["Teacher", "Training signal for a policy", "One-off training cost", "Needs judge accuracy"],
]


def _trajectory(rng, p_good):
    good = rng.random() < p_good
    return good


def simulate(judge_acc=0.8, actor_success=0.3, n=500, seed=0):
    """Synthetic: a judge labels n trajectories; a policy trained on the labels gets a
    success rate that grows with label accuracy. Illustration only."""
    try:
        judge_acc = min(max(float(judge_acc), 0.5), 1.0)
        actor_success = min(max(float(actor_success), 0.0), 1.0)
        n = int(min(max(int(n), 10), 5000))
        seed = int(seed)
    except (TypeError, ValueError):
        return "Enter numeric values.", []
    rng = random.Random(seed)
    labels_ok = 0
    for _ in range(n):
        good = _trajectory(rng, 0.5)
        said_good = good if rng.random() < judge_acc else (not good)
        labels_ok += int(said_good == good)
    label_acc = labels_ok / n
    student = actor_success + (0.9 - actor_success) * max(0.0, (label_acc - 0.5) / 0.5) * 0.8
    rows = [
        ["Decision model as actor", f"{actor_success:.0%}"],
        ["Judge label accuracy (sampled)", f"{label_acc:.0%}"],
        ["Policy taught by the judge", f"{student:.0%}"],
    ]
    msg = ("Simulated illustration, not the paper's results. A weak actor with an accurate "
           "trajectory judge can still teach a better policy.")
    return msg, rows


def role_advice(actor_success=0.3, judge_acc=0.8):
    try:
        a, j = float(actor_success), float(judge_acc)
    except (TypeError, ValueError):
        return "Enter numeric values."
    if j >= 0.75 and a < 0.5:
        return "Use as **teacher/judge**, not actor: it discriminates outcomes better than it acts."
    if a >= 0.5:
        return "Strong as actor; also check judge accuracy before using it as teacher."
    return "Weak at both roles; pick a different decision model."


IMPL_PATTERNS = [
    {"name": "Trajectory judge",
     "description": "Score the full episode once, then use the score as the RL reward.",
     "code": "def reward(trajectory, judge):\n    return judge(f'Did this episode succeed?\\n{trajectory}')\n"},
]
