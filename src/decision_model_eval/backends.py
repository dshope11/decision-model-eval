"""Backend registry: every model this project calls, with its pinned version.

All backends are reached through OpenRouter. Model ids are pinned to a version;
floating aliases (``-latest``) and routers (``jev-router``, which picks a model
per request) are never registered, since a measurement against a moving target
is not reproducible.

``provider`` pins the OpenRouter serving provider. Unpinned, OpenRouter routes
the same model id to different providers (Anthropic, Amazon Bedrock, Google
Vertex, Azure), and not every provider supports every feature - several Bedrock
endpoints for Opus 5.5 lack structured outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BackendInfo:
    """Metadata for one model backend."""

    model: str  # OpenRouter model id sent in the request
    resolved_model: str  # model id the response is expected to report
    api: Literal["chat", "decisions"]  # chat completions, or Jev's typed-question endpoint
    role: Literal["backend", "labeler"]  # backend under test, or the strong-model labeler
    provider: str  # OpenRouter provider tag to pin
    remote: bool  # True: a third party sees the payload, so private fields are stripped
    description: str


BACKENDS: dict[str, BackendInfo] = {
    "jev": BackendInfo(
        model="typesafe/jev-1.13",
        resolved_model="typesafe/jev-1.13-20260917",
        api="decisions",
        role="backend",
        provider="typesafe",
        remote=True,
        description="TypeSafe Jev 1.13, a decision model with typed questions",
    ),
    "haiku": BackendInfo(
        model="anthropic/claude-haiku-4.5",
        resolved_model="anthropic/claude-haiku-4.5",
        api="chat",
        role="backend",
        provider="anthropic",
        remote=True,
        description="Claude Haiku 4.5, a small general-purpose model",
    ),
    "opus": BackendInfo(
        model="anthropic/claude-opus-5.5",
        resolved_model="anthropic/claude-opus-5.5",
        api="chat",
        role="labeler",
        provider="anthropic",
        remote=True,
        description="Claude Opus 5.5, the strong-model labeler",
    ),
}


def backend_keys() -> list[str]:
    """Return the keys of all backends under test."""
    return [k for k, b in BACKENDS.items() if b.role == "backend"]


def labeler_keys() -> list[str]:
    """Return the keys of all labelers."""
    return [k for k, b in BACKENDS.items() if b.role == "labeler"]
