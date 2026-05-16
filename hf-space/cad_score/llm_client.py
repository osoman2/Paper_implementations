"""
Thin wrapper around the Anthropic API to generate N diverse responses
for a given prompt. Used by the Gradio demo.
"""

from __future__ import annotations
import os
import anthropic


def generate_responses(prompt: str, n: int = 5, model: str = "claude-haiku-4-5-20251001") -> list[str]:
    """
    Generate n responses to the same prompt.
    Each call is independent (no conversation context) to maximise diversity.
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    responses = []
    for _ in range(n):
        msg = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        responses.append(msg.content[0].text.strip())
    return responses


def generate_responses_high_temp(prompt: str, n: int = 5, model: str = "claude-haiku-4-5-20251001") -> list[str]:
    """Same as above but requests more creative/varied responses via system prompt."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    system = (
        "You are a highly creative assistant. Each response you give must take a "
        "completely different angle, framing, or perspective than what you've said before. "
        "Avoid repeating structures, metaphors, or ideas."
    )
    responses = []
    for _ in range(n):
        msg = client.messages.create(
            model=model,
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        responses.append(msg.content[0].text.strip())
    return responses
