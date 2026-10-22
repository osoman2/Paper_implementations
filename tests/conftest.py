import os
import signal
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPACE_DIR = ROOT / "hf-space"

# No telemetry or version pings from CI; keep matplotlib headless.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("MPLBACKEND", "Agg")

for path in (SPACE_DIR, ROOT / "tools", ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

IMPORT_TIMEOUT_S = int(os.environ.get("SMOKE_IMPORT_TIMEOUT", "120"))


@pytest.fixture(scope="session")
def app_module():
    """Import hf-space/app.py once, exactly as the Space does, with a hard timeout."""

    def _timeout(*_):
        raise TimeoutError(
            f"`import app` took more than {IMPORT_TIMEOUT_S}s: something heavy runs at import time "
            "(model download, network call). Load it lazily on first use instead."
        )

    previous = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(IMPORT_TIMEOUT_S)
    cwd = os.getcwd()
    try:
        os.chdir(SPACE_DIR)
        import app  # noqa: PLC0415

        return app
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        os.chdir(cwd)
