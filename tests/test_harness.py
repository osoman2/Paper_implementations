"""The smoke harness must catch the bug classes that reached production before."""

import gradio as gr

from smoke_harness import exercise


def test_catches_dataframe_rows_as_dicts():
    def bad_table():
        return [{"model": "a", "score": 1}]  # Dataframe wants rows as lists

    with gr.Blocks() as demo:
        btn = gr.Button()
        out = gr.Dataframe(headers=["model", "score"])
        btn.click(bad_table, None, out)
    report = exercise(demo, timeout_s=10)
    assert report.failures, "a Dataframe fed with dicts must be reported"


def test_catches_name_error_in_handler():
    def uses_missing_name(text: str):
        return Any  # noqa: F821 - the 2026-10-18 `typing.Any` incident

    with gr.Blocks() as demo:
        box = gr.Textbox()
        out = gr.Markdown()
        box.submit(uses_missing_name, box, out)
    report = exercise(demo, timeout_s=10)
    assert report.failures and "NameError" in report.failures[0].error


def test_uses_examples_as_second_case():
    seen = []

    def echo(text: str):
        seen.append(text)
        return text

    with gr.Blocks() as demo:
        box = gr.Textbox()
        out = gr.Markdown()
        gr.Examples([["hello"]], inputs=box)
        box.submit(echo, box, out)
    report = exercise(demo, timeout_s=10)
    assert not report.failures
    assert "" in seen and "hello" in seen


def test_clean_handler_passes():
    with gr.Blocks() as demo:
        n = gr.Slider(1, 5, value=2)
        out = gr.Dataframe(headers=["n", "sq"])
        n.change(lambda k: [[int(k), int(k) ** 2]], n, out)
    assert not exercise(demo, timeout_s=10).failures
