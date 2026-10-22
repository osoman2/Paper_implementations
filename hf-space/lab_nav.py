"""
Navigation for the Vector Ridge Lab Space.

Every POC tab is declared with `poc_tab("Label")` instead of `gr.Tab("Label")`. That gives it a
stable id (POC-001, POC-002, ... in declaration order, the same codes vectorridgelabs.com uses),
strips emojis from the visible label, and lets the region/POC selectors and `?poc=POC-050`
deep links select it.

New POCs: append a new `with poc_tab("Name: short claim"):` block at the end of the POC tabs in
app.py. Its region comes from REGION_BY_CODE in lab_regions.py when listed, otherwise from keywords in the label.
"""

import re
from pathlib import Path

import gradio as gr

from lab_regions import REGION_BY_CODE, REGIONS, region_for  # noqa: E402,F401

REGION_LABEL = dict(REGIONS)

_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF️‍⌚⌛⏩-⏺]"
)


def strip_emoji(text: str) -> str:
    return re.sub(r"\s{2,}", " ", _EMOJI.sub("", text)).strip()


def _index():
    """(code, label, region) for every poc_tab in app.py, in declaration order."""
    src = (Path(__file__).parent / "app.py").read_text(encoding="utf-8")
    labels = re.findall(r'^\s*with poc_tab\("([^"]+)"\)', src, re.M)
    out = []
    for i, raw in enumerate(labels, start=1):
        code = f"POC-{i:03d}"
        label = strip_emoji(raw)
        out.append((code, label, region_for(code, label)))
    return out


INDEX = _index()
CODES = {code for code, _, _ in INDEX}
_counter = 0


def poc_tab(label: str) -> gr.Tab:
    global _counter
    _counter += 1
    code = f"POC-{_counter:03d}"
    return gr.Tab(strip_emoji(label), id=code, elem_id=code.lower())


def poc_choices(region: str | None):
    rows = [r for r in INDEX if not region or region == "all" or r[2] == region]
    return [(f"{code} · {label}", code) for code, label, _ in rows]


REGION_CHOICES = [("All regions", "all")] + [(label, rid) for rid, label in REGIONS]


THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.zinc,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_sm,
).set(
    body_background_fill_dark="#0b0b0d",
    background_fill_primary_dark="#111114",
    background_fill_secondary_dark="#16161a",
    block_background_fill_dark="#111114",
    block_border_color_dark="#26262b",
    border_color_primary_dark="#26262b",
    button_primary_background_fill_dark="#f97316",
    button_primary_background_fill_hover_dark="#ea580c",
    button_primary_text_color_dark="#0b0b0d",
)

CSS = """
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; }
#poc-tabs > .tab-nav { display: none; }
#vrl-header { border-bottom: 1px solid #26262b; padding-bottom: 14px; margin-bottom: 4px; }
#vrl-header h1 { font-weight: 600; letter-spacing: -0.01em; margin: 0; }
#vrl-header p { color: #a1a1aa; margin: 6px 0 0; }
#vrl-header .eyebrow { font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.28em;
  text-transform: uppercase; color: #f97316; }
#vrl-nav { gap: 12px; }
"""

# The Lab is dark-only, like vectorridgelabs.com. Reload once with Gradio's dark theme flag,
# keeping any ?poc= deep link.
FORCE_DARK_JS = """
() => {
  const url = new URL(window.location.href);
  if (url.searchParams.get('__theme') !== 'dark') {
    url.searchParams.set('__theme', 'dark');
    window.location.replace(url.href);
  }
}
"""

HEADER_MD = (
    '<div id="vrl-header">'
    '<span class="eyebrow">Vector Ridge Lab</span>'
    "<h1>Proofs of concept from recent AI papers</h1>"
    "<p>Each POC is a runnable reproduction of a technique we track for production AI systems. "
    'Pick a region and a POC, or open this page from <a href="https://vectorridgelabs.com/en/lab" '
    'target="_blank">vectorridgelabs.com/lab</a>.</p>'
    "</div>"
)


def wire_navigation(demo: gr.Blocks, region_dd: gr.Dropdown, poc_dd: gr.Dropdown, tabs: gr.Tabs) -> None:
    def on_region(region):
        return gr.Dropdown(choices=poc_choices(region), value=None)

    def on_poc(code):
        if code in CODES:
            return gr.Tabs(selected=code)
        return gr.Tabs()

    def on_load(request: gr.Request):
        code = (request.query_params.get("poc") or "").strip().upper() if request else ""
        if code in CODES:
            region = next(r for c, _, r in INDEX if c == code)
            return (
                gr.Dropdown(value=region),
                gr.Dropdown(choices=poc_choices(region), value=code),
                gr.Tabs(selected=code),
            )
        return gr.Dropdown(), gr.Dropdown(), gr.Tabs()

    region_dd.input(on_region, inputs=region_dd, outputs=poc_dd)
    poc_dd.change(on_poc, inputs=poc_dd, outputs=tabs)
    demo.load(on_load, inputs=None, outputs=[region_dd, poc_dd, tabs])
