"""
NemoClaw — Infrastructure layer sandbox simulation.

Models the OS-level security sandbox that separates the control plane
from the agent execution environment:
  - Landlock filesystem confinement (allowlist of r/w directories)
  - seccomp syscall filter (deny dangerous kernel operations)
  - Network namespace egress control (block unapproved sockets)
  - Credential injection gateway (agent never holds real API keys)
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# File-system confinement (Landlock simulation)
# ---------------------------------------------------------------------------

ALLOWED_READ = ["/workspace", "/tmp/agent-scratch", "/usr/lib", "/etc/ssl/certs"]
ALLOWED_WRITE = ["/workspace/output", "/tmp/agent-scratch"]

SENSITIVE_PATHS = [
    "/etc/passwd", "/etc/shadow", "~/.ssh", "~/.aws/credentials",
    "/proc/self", "/sys/kernel", "/root/.config",
    "/home/*/.ssh", "/var/run/secrets",
]

def check_filesystem(path: str) -> dict:
    """Return sandbox verdict for a requested file-system path."""
    path = path.strip()
    for p in SENSITIVE_PATHS:
        pattern = p.replace("*", "[^/]+").replace("~", "/root")
        if re.search(pattern, path):
            return {
                "allowed": False,
                "layer": "Landlock (NemoClaw)",
                "reason": f"Path matches sensitive-file blocklist pattern `{p}`",
                "mitigation": "Agent process receives EPERM; action never reaches host OS.",
            }
    for allowed in ALLOWED_READ:
        if path.startswith(allowed):
            return {
                "allowed": True,
                "layer": "Landlock (NemoClaw)",
                "reason": f"Path is within allowlisted read directory `{allowed}`",
                "mitigation": None,
            }
    return {
        "allowed": False,
        "layer": "Landlock (NemoClaw)",
        "reason": "Path is outside all allowlisted directories",
        "mitigation": "Agent process receives EPERM; only /workspace and /tmp/agent-scratch are accessible.",
    }


# ---------------------------------------------------------------------------
# Syscall filter (seccomp simulation)
# ---------------------------------------------------------------------------

BLOCKED_SYSCALLS = {
    "ptrace": "Can be used for sandbox escape via process tracing",
    "mount": "Could remount filesystems to bypass Landlock",
    "init_module": "Kernel module loading could install rootkits",
    "create_module": "Deprecated but dangerous kernel interface",
    "kexec_load": "Could replace the running kernel",
    "pivot_root": "Could escape the container filesystem namespace",
    "unshare": "Could create new namespaces to escape confinement",
    "setns": "Could join namespaces and escape isolation",
    "chroot": "Classic container escape vector",
    "sethostname": "Signals namespace boundary violation",
}

ALLOWED_SYSCALLS_EXAMPLES = [
    "read", "write", "open", "close", "stat", "mmap", "munmap",
    "brk", "futex", "clone", "execve", "exit", "socket", "connect",
]

def check_syscall(syscall: str) -> dict:
    syscall = syscall.strip().lower()
    if syscall in BLOCKED_SYSCALLS:
        return {
            "allowed": False,
            "layer": "seccomp (NemoClaw)",
            "reason": BLOCKED_SYSCALLS[syscall],
            "mitigation": "Kernel returns EPERM/ENOSYS immediately; agent cannot escalate privileges.",
        }
    if syscall in ALLOWED_SYSCALLS_EXAMPLES:
        return {
            "allowed": True,
            "layer": "seccomp (NemoClaw)",
            "reason": "Syscall is on the seccomp allowlist for normal agent operation",
            "mitigation": None,
        }
    return {
        "allowed": False,
        "layer": "seccomp (NemoClaw)",
        "reason": "Syscall is not on the explicit allowlist (default-deny)",
        "mitigation": "Default-deny seccomp profile blocks any unlisted syscall.",
    }


# ---------------------------------------------------------------------------
# Credential gateway
# ---------------------------------------------------------------------------

def simulate_credential_request(endpoint: str, placeholder_key: str) -> dict:
    """
    NemoClaw intercepts egress traffic and injects real credentials
    only after the destination is approved. The agent only ever sees
    placeholder values.
    """
    known_approved = [
        "api.openai.com", "api.anthropic.com",
        "api.pinecone.io", "api.cohere.com",
    ]
    host = re.sub(r"https?://", "", endpoint).split("/")[0]
    approved = host in known_approved
    return {
        "endpoint": endpoint,
        "host": host,
        "placeholder_used": placeholder_key,
        "gateway_decision": "APPROVED — inject real credential" if approved else "BLOCKED — requires human approval",
        "real_key_visible_to_agent": False,
        "approved": approved,
        "note": (
            "Real API key stays on host machine. "
            "Gateway intercepts the TLS-layer request and injects it transparently."
            if approved else
            "New endpoint detected. Human operator must approve before credential injection occurs."
        ),
    }


# ---------------------------------------------------------------------------
# Threat scenario library
# ---------------------------------------------------------------------------

THREAT_SCENARIOS = {
    "Read SSH keys": {
        "type": "filesystem",
        "input": "/root/.ssh/id_rsa",
        "description": "Agent tries to read host SSH private key (prompt injection)",
    },
    "Load kernel module": {
        "type": "syscall",
        "input": "init_module",
        "description": "Agent attempts to load a kernel module (sandbox escape)",
    },
    "Access /etc/shadow": {
        "type": "filesystem",
        "input": "/etc/shadow",
        "description": "Agent reads password hash file (credential theft)",
    },
    "Use ptrace for sandbox escape": {
        "type": "syscall",
        "input": "ptrace",
        "description": "Agent tries to trace another process to escape sandbox",
    },
    "Read workspace output (allowed)": {
        "type": "filesystem",
        "input": "/workspace/output/report.txt",
        "description": "Normal agent output — should be allowed",
    },
    "Write to scratch space (allowed)": {
        "type": "filesystem",
        "input": "/tmp/agent-scratch/temp_data.json",
        "description": "Normal agent scratch write — should be allowed",
    },
}
