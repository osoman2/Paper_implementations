"""
NightWatch: proactive perception for computer-use agents (Tab 58)
Source: Hugging Face Daily Papers · 2026-11-25 · arXiv 2611.28947
Score: 8/10

The paper adds a lightweight streaming watcher to a vision-language computer-use agent so that its
perception stays synchronized with a dynamic interface, cutting latency and wasted waiting while
keeping or improving task success.

This demo is an ILLUSTRATION on a simulated UI (random page-settle times). It is not the paper's
method, model or data; it only shows why a fixed-wait agent either wastes time or acts on a stale screen.
"""

from __future__ import annotations

import numpy as np

FINDINGS = [
    {"name": "Fixed waits trade time for staleness",
     "body": "An agent that acts, waits a fixed time and then looks either idles after the screen "
             "settled or looks too early and reasons over a stale screenshot."},
    {"name": "A cheap watcher decouples perception from reasoning",
     "body": "A lightweight stream checks the screen for change and wakes the expensive model only "
             "when the interface has settled, so waiting costs almost nothing and observations are fresh."},
    {"name": "Illustration only",
     "body": "All numbers in this tab come from a simulated interface, not from the paper."},
]

IMPL_PATTERNS = [
    {"name": "Fixed-wait loop (baseline)",
     "description": "Sleep a constant time after each action, then take one screenshot.",
     "code": "def step(agent, ui, wait):\n    agent.act()\n    time.sleep(wait)\n    return agent.observe(ui.screenshot())"},
    {"name": "Streaming watcher",
     "description": "Poll a cheap frame-difference check; call the model only after the screen is stable.",
     "code": ("def step(agent, ui, poll=0.2, stable=2):\n    agent.act()\n    prev, calm = None, 0\n"
              "    while calm < stable:\n        time.sleep(poll)\n        frame = ui.thumbnail()\n"
              "        calm = calm + 1 if frame == prev else 0\n        prev = frame\n"
              "    return agent.observe(ui.screenshot())")},
]


def _validate(mean_settle, wait, poll, reason, stale_err, n_steps, seed):
    try:
        m, w, p, r, q, n, s = (float(mean_settle), float(wait), float(poll), float(reason),
                               float(stale_err), int(n_steps), int(seed))
    except (TypeError, ValueError):
        return None
    if not (0.1 <= m <= 20 and 0.1 <= w <= 30 and 0.05 <= p <= 5 and 0.1 <= r <= 20):
        return None
    if not 0.0 <= q <= 1.0 or not 10 <= n <= 20000:
        return None
    return m, w, p, r, q, n, s


def simulate(mean_settle=2.0, wait=2.0, poll=0.25, reason=3.0, stale_err=0.3, n_steps=2000, seed=0):
    """Return (markdown, rows). rows: [strategy, seconds per step, idle seconds per step, failed steps (%)]."""
    args = _validate(mean_settle, wait, poll, reason, stale_err, n_steps, seed)
    if args is None:
        return ("Use settle 0.1-20 s, wait 0.1-30 s, poll 0.05-5 s, reasoning 0.1-20 s, "
                "stale error 0-1 and 10-20000 steps.", [])
    m, w, p, r, q, n, s = args
    rng = np.random.default_rng(s)
    settle = rng.exponential(m, size=n)
    # Fixed wait: look at time w. If the screen has not settled, the observation is stale and the
    # model errs with probability q, otherwise it spots the problem and waits again (one retry).
    idle_f = np.where(settle <= w, w - settle, 0.0)
    stale = settle > w
    err = rng.random(n) < q
    fail_f = stale & err
    retry = stale & ~err
    time_f = w + r + np.where(retry, w + r, 0.0)
    idle_f = idle_f + np.where(retry, np.maximum(2 * w - (settle), 0.0), 0.0)
    # Watcher: noticed at the next poll tick after settling, model runs once on a fresh frame.
    seen = np.ceil(settle / p) * p
    time_w = seen + r
    idle_w = seen - settle
    rows = [
        ["Fixed wait", round(float(time_f.mean()), 2), round(float(idle_f.mean()), 2),
         round(100 * float(fail_f.mean()), 1)],
        ["Streaming watcher", round(float(time_w.mean()), 2), round(float(idle_w.mean()), 2), 0.0],
    ]
    saved = 100 * (1 - time_w.mean() / time_f.mean())
    return (f"Watcher saves {saved:.0f}% of time per step and avoids stale-screen failures "
            f"(simulated, illustration only).", rows)


def sweep_wait(mean_settle=2.0, poll=0.25, reason=3.0, stale_err=0.3, n_steps=2000, seed=0):
    """Return (markdown, rows) for several fixed waits against one watcher."""
    rows = []
    for w in (0.5, 1, 2, 4, 8):
        _, r2 = simulate(mean_settle, w, poll, reason, stale_err, n_steps, seed)
        if not r2:
            return ("Invalid parameters.", [])
        rows.append([w, r2[0][1], r2[0][3], r2[1][1]])
    return ("Short waits fail on stale screens; long waits idle. The watcher avoids both "
            "(simulated, illustration only).", rows)
