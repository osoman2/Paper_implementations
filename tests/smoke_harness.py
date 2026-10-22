"""
Generic event-handler exerciser for Gradio Blocks.

For every event handler registered in a Blocks app it calls the Python function with the values
the browser would send (each input component's default, or the first row of a gr.Examples set
that targets those inputs), then runs every returned value through the output component's
`postprocess`. That is the path that broke in production when a gr.Dataframe received a list of
dicts instead of a list of rows: the function itself ran fine, the component rejected its output.

No test per tab is needed: a new POC tab is covered as soon as it is wired with `.click`/`.change`.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import re
import signal
import time
from dataclasses import dataclass, field

import gradio as gr

# Gradio passes these frontend-only objects through untouched.
_PASSTHROUGH_TYPES = (gr.components.Component,)


class HandlerTimeout(Exception):
    pass


# Errors that only mean "this sandbox cannot reach the model hub". With SMOKE_ALLOW_OFFLINE=1
# (agent sandboxes behind an allowlist proxy) they are reported as skipped instead of failed.
# CI runs with network and keeps them as failures.
_OFFLINE_ERROR = re.compile(
    r"couldn't connect to 'https://huggingface\.co'|Tunnel connection failed|ProxyError|"
    r"Temporary failure in name resolution|Network is unreachable",
    re.I,
)


@dataclass
class HandlerResult:
    index: int
    name: str
    case: str
    ok: bool
    seconds: float
    error: str = ""
    skipped: bool = False


@dataclass
class Report:
    results: list[HandlerResult] = field(default_factory=list)

    @property
    def failures(self) -> list[HandlerResult]:
        return [r for r in self.results if not r.ok and not r.skipped]

    @property
    def skipped(self) -> list[HandlerResult]:
        return [r for r in self.results if r.skipped]

    def summary(self) -> str:
        lines = [
            f"{len(self.results)} handler calls, {len(self.failures)} failed, "
            f"{len(self.skipped)} skipped (offline)"
        ]
        for r in self.failures:
            lines.append(f"  #{r.index} {r.name} [{r.case}] {r.error}")
        for r in self.skipped:
            lines.append(f"  offline: #{r.index} {r.name} [{r.case}] {r.error[:120]}")
        slow = sorted((r for r in self.results if r.ok and r.seconds > 5), key=lambda r: -r.seconds)
        for r in slow[:5]:
            lines.append(f"  slow: #{r.index} {r.name} [{r.case}] {r.seconds:.1f}s")
        return "\n".join(lines)


def _browser_default(component) -> object:
    value = getattr(component, "value", None)
    # An empty textbox is sent as "" by the browser, not None.
    if value is None and isinstance(component, (gr.Textbox, gr.Code)):
        return ""
    return value


def _example_rows(blocks: gr.Blocks) -> dict[int, list]:
    """component id -> first example value, from every gr.Examples / gr.Dataset in the app."""
    rows: dict[int, list] = {}
    for block in blocks.blocks.values():
        if isinstance(block, gr.Dataset) and getattr(block, "raw_samples", None):
            first = block.raw_samples[0]
            for comp, val in zip(block._components, first):
                rows.setdefault(comp._id, val)
    return rows


def _run(fn, args, timeout_s: int):
    def _alarm(*_):
        raise HandlerTimeout(f"no result after {timeout_s}s")

    previous = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout_s)
    try:
        out = fn(*args)
        if inspect.iscoroutine(out):
            out = asyncio.run(out)
        if inspect.isgenerator(out):
            last = None
            for last in out:
                pass
            out = last
        return out
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _postprocess(outputs, out) -> None:
    if not outputs:
        return
    values = [out] if len(outputs) == 1 else list(out) if isinstance(out, (tuple, list)) else [out]
    if len(outputs) > 1 and len(values) != len(outputs):
        raise ValueError(f"handler returned {len(values)} values for {len(outputs)} outputs")
    for comp, val in zip(outputs, values):
        if isinstance(val, _PASSTHROUGH_TYPES):
            continue
        if isinstance(val, dict) and val.get("__type__") == "update":
            continue
        if hasattr(comp, "postprocess"):
            comp.postprocess(val)


def _skip(fn) -> bool:
    # gr.Examples loaders and request-aware navigation hooks are Gradio/frontend plumbing.
    if inspect.iscoroutinefunction(fn):
        return True
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True
    return "request" in params


def exercise(blocks: gr.Blocks, timeout_s: int = 60, allow_offline: bool | None = None) -> Report:
    if allow_offline is None:
        allow_offline = os.environ.get("SMOKE_ALLOW_OFFLINE") == "1"
    report = Report()
    fns = blocks.fns.values() if isinstance(blocks.fns, dict) else blocks.fns
    examples = _example_rows(blocks)
    for index, block_fn in enumerate(fns):
        fn = block_fn.fn
        if fn is None or _skip(fn):
            continue
        name = getattr(fn, "__name__", repr(fn))
        cases = {"defaults": [_browser_default(c) for c in block_fn.inputs]}
        if any(c._id in examples for c in block_fn.inputs):
            cases["example"] = [examples.get(c._id, _browser_default(c)) for c in block_fn.inputs]
        for case, args in cases.items():
            start = time.time()
            try:
                out = _run(fn, args, timeout_s)
                _postprocess(block_fn.outputs, out)
                report.results.append(HandlerResult(index, name, case, True, time.time() - start))
            except Exception as exc:  # noqa: BLE001 - every failure is reported, none is fatal here
                msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:240] if str(exc) else ''}"
                offline = allow_offline and bool(_OFFLINE_ERROR.search(f"{exc!r} {exc}"))
                report.results.append(
                    HandlerResult(index, name, case, False, time.time() - start, msg, skipped=offline)
                )
    return report
