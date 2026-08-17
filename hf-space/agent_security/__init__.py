"""
AI Agent Security — 3-Layer Defense Stack POC.

Based on: "The 3-layer security stack for AI agents"
(AlphaSignal deep-dive · 2026-08-16)
Covers: NemoClaw (infrastructure), NanoClaw (architecture/runtime), CrabTrap (network)

Architecture:
  Layer 1 — Infrastructure (NemoClaw):
    OS-level sandbox: Landlock filesystem confinement + seccomp syscall filtering
    + network namespaces + credential-injection gateway (agent never holds real keys)

  Layer 2 — Architecture & Runtime (NanoClaw):
    Minimalist ephemeral containers: codebase < 5K LOC vs 1M+ LOC for full stack
    Echo-hardened runtime: continuously rebuilds software to strip known CVEs

  Layer 3 — Network (CrabTrap by Brex):
    Bifurcated proxy: static rules for low-risk requests, LLM-as-judge for high-risk
    Human-in-the-loop escalation; agent cannot send unapproved outbound traffic
"""

from .nemoclaw import (
    check_filesystem,
    check_syscall,
    simulate_credential_request,
    THREAT_SCENARIOS as NEMOCLAW_SCENARIOS,
    ALLOWED_READ,
    ALLOWED_WRITE,
    SENSITIVE_PATHS,
    BLOCKED_SYSCALLS,
    ALLOWED_SYSCALLS_EXAMPLES,
)

from .nanoclaw import (
    get_attack_surface_comparison,
    simulate_session_lifecycle,
    PDF_EXPLOIT_SCENARIOS,
    FULL_OPENCLAW_COMPONENTS,
    NANOCLAW_KEPT,
)

from .crabtrap import (
    evaluate_request,
    DEMO_SCENARIOS as CRABTRAP_SCENARIOS,
)

__all__ = [
    "check_filesystem",
    "check_syscall",
    "simulate_credential_request",
    "NEMOCLAW_SCENARIOS",
    "ALLOWED_READ",
    "ALLOWED_WRITE",
    "SENSITIVE_PATHS",
    "BLOCKED_SYSCALLS",
    "ALLOWED_SYSCALLS_EXAMPLES",
    "get_attack_surface_comparison",
    "simulate_session_lifecycle",
    "PDF_EXPLOIT_SCENARIOS",
    "FULL_OPENCLAW_COMPONENTS",
    "NANOCLAW_KEPT",
    "evaluate_request",
    "CRABTRAP_SCENARIOS",
]
