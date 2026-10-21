"""Backend registry: every model this project calls, with its pinned version.

All backends are reached through OpenRouter. Model ids are pinned to a version;
floating aliases (``-latest``) and routers (``jev-router``, which picks a model
per request) are never registered, since a measurement against a moving target
is not reproducible.

``provider`` pins the OpenRouter serving provider. Unpinned, OpenRouter routes
the same model id to different providers (Anthropic, Amazon Bedrock, Google
Vertex, Azure), and not every provider supports every feature - several Bedrock
endpoints for Opus 5.5 lack structured outputs.

``effort`` pins the reasoning effort, which changes the answers as much as a model
version does, so it is part of the recorded model identity (``model_label``). Anthropic
models go through OpenRouter's Anthropic-format endpoint (``api="messages"``) because it
is the route where the setting reaches the model: on 2026-10-20, ``output_config.effort``
there raised Haiku 5.5's thinking tokens steadily from low to max, while
``reasoning.effort`` on chat completions had no measurable effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BackendInfo:
    """Metadata for one model backend."""

    model: str  # OpenRouter model id sent in the request
    resolved_model: str  # model id the response is expected to report
    api: Literal["messages", "decisions"]  # Anthropic-format messages, or Jev's typed questions
    role: Literal["backend", "labeler"]  # backend under test, or the strong-model labeler
    provider: str  # OpenRouter provider tag to pin
    remote: bool  # True: a third party sees the payload, so private fields are stripped
    effort: str | None  # pinned reasoning effort; None where the model has no such setting
    description: str

    @property
    def model_label(self) -> str:
        """The model as recorded: the resolved id, plus ``/effort=<level>`` where pinned."""
        if self.effort is None:
            return self.resolved_model
        return f"{self.resolved_model}/effort={self.effort}"


BACKENDS: dict[str, BackendInfo] = {
    "jev": BackendInfo(
        model="typesafe/jev-1.14",
        resolved_model="typesafe/jev-1.14-20261014",
        api="decisions",
        role="backend",
        provider="typesafe",
        remote=True,
        effort=None,
        description="TypeSafe Jev 1.14, a decision model with typed questions",
    ),
    "haiku": BackendInfo(
        model="anthropic/claude-haiku-5.5",
        resolved_model="anthropic/claude-haiku-5.5",
        api="messages",
        role="backend",
        provider="anthropic",
        remote=True,
        effort="medium",  # the model's default, pinned so a changed default can't move results
        description="Claude Haiku 5.5, a small general-purpose model",
    ),
    "opus": BackendInfo(
        model="anthropic/claude-opus-5.5",
        resolved_model="anthropic/claude-opus-5.5",
        api="messages",
        role="labeler",
        provider="anthropic",
        remote=True,
        effort=None,  # chosen at M6 with the labeling prompts; decide() refuses it until then
        description="Claude Opus 5.5, the strong-model labeler",
    ),
}


def backend_keys() -> list[str]:
    """Return the keys of all backends under test."""
    return [k for k, b in BACKENDS.items() if b.role == "backend"]


def labeler_keys() -> list[str]:
    """Return the keys of all labelers."""
    return [k for k, b in BACKENDS.items() if b.role == "labeler"]
