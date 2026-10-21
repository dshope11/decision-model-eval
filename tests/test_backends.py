"""Tests for the backend registry."""

from __future__ import annotations

import pytest

from decision_model_eval.backends import BACKENDS, backend_keys, labeler_keys


@pytest.mark.parametrize("key", list(BACKENDS))
def test_model_ids_are_pinned(key: str) -> None:
    """No floating alias or router may be registered - results must be reproducible."""
    info = BACKENDS[key]
    for model_id in (info.model, info.resolved_model):
        assert "latest" not in model_id
        assert "router" not in model_id


@pytest.mark.parametrize("key", list(BACKENDS))
def test_resolved_model_extends_requested_model(key: str) -> None:
    info = BACKENDS[key]
    assert info.resolved_model.startswith(info.model)


def test_roles() -> None:
    assert backend_keys() == ["jev", "haiku"]
    assert labeler_keys() == ["opus"]


def test_only_jev_uses_decisions_api() -> None:
    assert [k for k, b in BACKENDS.items() if b.api == "decisions"] == ["jev"]


def test_every_messages_backend_under_test_pins_its_effort() -> None:
    """Effort changes the answers as much as a model version, so it is never left default."""
    for key in backend_keys():
        info = BACKENDS[key]
        if info.api == "messages":
            assert info.effort in {"low", "medium", "high", "xhigh", "max"}


def test_model_label_carries_the_effort() -> None:
    assert BACKENDS["haiku"].model_label == "anthropic/claude-haiku-5.5/effort=medium"
    assert BACKENDS["jev"].model_label == "typesafe/jev-1.14-20261014"
