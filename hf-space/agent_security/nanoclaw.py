"""
NanoClaw — Architecture & runtime layer simulation.

Models the minimalist, single-process, ephemeral container approach:
  - Each agent session gets its own isolated container
  - Codebase is reduced from millions of lines → auditable few thousand
  - Echo-hardened runtime: known CVEs are continuously rebuilt away
  - Attack surface scoring: counts components and known CVE exposure
"""

from __future__ import annotations
import random
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Simulated software components and their CVE exposure
# ---------------------------------------------------------------------------

@dataclass
class Component:
    name: str
    lines_of_code: int
    known_cves: int
    category: str
    removable: bool = True
    reason_kept: str = ""


FULL_OPENCLAW_COMPONENTS = [
    Component("PDF parser (pdfminer)",          45_000,  7, "document-handling"),
    Component("Web browser engine (Chromium)",  312_000, 18, "web"),
    Component("Video decoder (ffmpeg)",         98_000, 12, "media"),
    Component("Image library (Pillow/libjpeg)", 22_000,  4, "media"),
    Component("Office suite parser (python-docx)",15_000, 3, "document-handling"),
    Component("Audio codec (libopus)",           8_500,  1, "media"),
    Component("SMTP client (smtplib+deps)",      6_200,  2, "network"),
    Component("SSH library (paramiko)",         18_000,  5, "network"),
    Component("Database ORM (SQLAlchemy)",       35_000, 2, "data"),
    Component("Kubernetes client (k8s-python)", 42_000,  6, "orchestration"),
    Component("Docker SDK",                     28_000,  4, "orchestration"),
    Component("Legacy SOAP adapter",            11_000,  8, "legacy"),
    Component("JWT auth stack",                  4_200,  3, "auth"),
    Component("Agent core (tool dispatch)",      3_800,  0, "core", removable=False,
              reason_kept="Required for agent operation"),
    Component("LLM client (minimal HTTP)",       1_200,  0, "core", removable=False,
              reason_kept="Required for LLM API calls"),
    Component("File I/O helpers",                  800,  0, "core", removable=False,
              reason_kept="Required for task execution"),
]

NANOCLAW_KEPT = [c for c in FULL_OPENCLAW_COMPONENTS if not c.removable]


def get_attack_surface_comparison() -> dict:
    full_loc = sum(c.lines_of_code for c in FULL_OPENCLAW_COMPONENTS)
    full_cves = sum(c.known_cves for c in FULL_OPENCLAW_COMPONENTS)
    full_components = len(FULL_OPENCLAW_COMPONENTS)

    nano_loc = sum(c.lines_of_code for c in NANOCLAW_KEPT)
    nano_cves = sum(c.known_cves for c in NANOCLAW_KEPT)
    nano_components = len(NANOCLAW_KEPT)

    removed = [c for c in FULL_OPENCLAW_COMPONENTS if c.removable]

    return {
        "full": {
            "name": "OpenClaw (full)",
            "components": full_components,
            "lines_of_code": full_loc,
            "known_cves": full_cves,
        },
        "nano": {
            "name": "NanoClaw (minimal)",
            "components": nano_components,
            "lines_of_code": nano_loc,
            "known_cves": nano_cves,
        },
        "removed_components": [
            {"name": c.name, "loc": c.lines_of_code, "cves": c.known_cves, "category": c.category}
            for c in removed
        ],
        "reduction": {
            "loc_pct": round((1 - nano_loc / full_loc) * 100, 1),
            "cve_pct": round((1 - nano_cves / full_cves) * 100 if full_cves else 0, 1),
            "components_pct": round((1 - nano_components / full_components) * 100, 1),
        },
    }


# ---------------------------------------------------------------------------
# Ephemeral container simulation
# ---------------------------------------------------------------------------

def simulate_session_lifecycle(task: str) -> list[dict]:
    """Simulate the birth-and-death of an ephemeral NanoClaw container."""
    session_id = f"nanoclaw-{random.randint(10000, 99999)}"
    return [
        {"step": 1, "event": "SPAWN",
         "detail": f"New ephemeral container `{session_id}` created from hardened base image"},
        {"step": 2, "event": "HARDEN",
         "detail": "Echo rebuilds runtime: PDF parser CVE-2026-1234 patched, chromium stripped, SSH removed"},
        {"step": 3, "event": "ISOLATE",
         "detail": "Network namespace, Landlock rules, and seccomp profile applied"},
        {"step": 4, "event": "EXECUTE",
         "detail": f"Agent runs task: '{task[:80]}'"},
        {"step": 5, "event": "AUDIT",
         "detail": "All syscalls logged; filesystem diff recorded; outbound connections verified"},
        {"step": 6, "event": "DESTROY",
         "detail": f"Container `{session_id}` destroyed — no state persists between sessions"},
    ]


# ---------------------------------------------------------------------------
# CVE injection attack simulation (Echo-hardened vs standard)
# ---------------------------------------------------------------------------

PDF_EXPLOIT_SCENARIOS = [
    {
        "name": "Malicious PDF with CVE-2026-1234 payload",
        "cve": "CVE-2026-1234",
        "component": "PDF parser (pdfminer)",
        "standard_outcome": "🔴 COMPROMISED — payload executes inside agent process via heap overflow",
        "echo_outcome": "✅ BLOCKED — Echo rebuilt pdfminer without the vulnerable codepath; payload lands on dead code",
    },
    {
        "name": "Web page with Chromium zero-day (CVE-2026-5678)",
        "cve": "CVE-2026-5678",
        "component": "Web browser engine (Chromium)",
        "standard_outcome": "🔴 COMPROMISED — renderer exploit achieves RCE inside agent container",
        "echo_outcome": "✅ BLOCKED — Chromium is not present in NanoClaw; agent uses headless text-only fetch",
    },
    {
        "name": "Office document macro exploit (CVE-2026-9012)",
        "cve": "CVE-2026-9012",
        "component": "Office suite parser (python-docx)",
        "standard_outcome": "🔴 COMPROMISED — malicious OOXML triggers code exec during parse",
        "echo_outcome": "✅ BLOCKED — python-docx removed from runtime; task fails gracefully with error message",
    },
]
