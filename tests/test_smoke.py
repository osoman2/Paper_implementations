"""
Smoke tests for the Hugging Face Space (hf-space/app.py).

These run on every pull request into the deploy branch and must pass before merge.
They exist because the first full deploy (2026-10-18) failed on bugs that no single
agent run could see: a missing import in app.py, a tab feeding dicts to gr.Dataframe,
and dependency drift (gradio 4.44 vs huggingface_hub 1.x / starlette 1.x).
"""

import re
import socket
import sys
from pathlib import Path

import pytest

SPACE = Path(__file__).resolve().parent.parent / "hf-space"
sys.path.insert(0, str(SPACE))

LOCKED_PINS = {
    "gradio": "==4.44.0",
    "huggingface_hub": "<1.0",
    "starlette": "<1.0",
}


@pytest.fixture(scope="session")
def app_module():
    import app  # noqa: E402  (NameError / ImportError at import time fails here)

    return app


def test_locked_pins_unchanged():
    req = (SPACE / "requirements.txt").read_text()
    for pkg, spec in LOCKED_PINS.items():
        line = next((l for l in req.splitlines() if re.match(rf"^{re.escape(pkg)}\b", l)), None)
        assert line, f"{pkg} missing from requirements.txt"
        assert spec in line, f"{pkg} pin changed: '{line}' (must contain '{spec}')"


def test_app_imports_and_builds(app_module):
    import gradio as gr

    assert isinstance(app_module.demo, gr.Blocks)


def test_every_poc_tab_registered(app_module):
    import lab_nav

    declared = len(re.findall(r'^\s*with poc_tab\("', (SPACE / "app.py").read_text(), re.M))
    assert declared == len(lab_nav.INDEX), "poc_tab index out of sync with app.py"
    assert lab_nav._counter == declared, (
        f"{declared} poc_tab blocks declared but {lab_nav._counter} built: a tab block failed to execute"
    )


# Tabs 1-51 predate the "Name: short claim" convention; every tab after them must follow it.
LEGACY_TABS = 51
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF]")


def test_new_tab_labels_follow_convention():
    labels = re.findall(r'^\s*with poc_tab\("([^"]+)"\)', (SPACE / "app.py").read_text(), re.M)
    for label in labels[LEGACY_TABS:]:
        assert ":" in label, f"tab label must be 'Name: short claim': {label!r}"
        assert not _EMOJI.search(label), f"no emojis in tab labels: {label!r}"


def test_space_serves_http(app_module):
    import httpx

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    demo = app_module.demo
    demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True)
    try:
        base = f"http://127.0.0.1:{port}"
        r = httpx.get(base + "/", timeout=30)
        assert r.status_code == 200, f"/ returned {r.status_code}"
        cfg = httpx.get(base + "/config", timeout=30)
        assert cfg.status_code == 200, f"/config returned {cfg.status_code}"
        assert len(cfg.json().get("components", [])) > 50
    finally:
        demo.close()


def test_event_handlers_run_with_defaults(app_module):
    """Run every event handler whose inputs all have defaults, through Gradio's real
    preprocess -> fn -> postprocess path, exactly as a click in the browser would.

    This is what catches output-shape bugs (e.g. dict rows sent to gr.Dataframe), which
    only fail in postprocess. Handlers that need typed user input or example rows are skipped.
    """
    import asyncio

    import gradio as gr
    from gradio.state_holder import SessionState

    demo = app_module.demo
    failures, checked = [], 0

    async def run_all():
        nonlocal checked
        for block_fn in demo.fns.values():
            raw = [getattr(c, "value", None) for c in block_fn.inputs]
            if (
                block_fn.fn is None
                or any(isinstance(c, gr.Dataset) for c in block_fn.inputs)
                or any(v in (None, "") for v in raw)
            ):
                continue
            try:
                await asyncio.wait_for(
                    demo.process_api(block_fn=block_fn, inputs=raw, state=SessionState(demo)), 120
                )
                checked += 1
            except Exception as e:
                root = e.__cause__ or e
                name = getattr(block_fn.fn, "__name__", "?")
                failures.append(f"{name}: {type(root).__name__}: {str(root)[:300]}")

    asyncio.run(run_all())
    assert not failures, f"{len(failures)} handler(s) crash on default inputs:\n" + "\n".join(failures)
    assert checked >= 50, f"only {checked} handlers exercised; the harness is not reaching the tabs"
