"""Tests for decision_model_eval.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from decision_model_eval.config import REPO_ROOT, EvalConfig, load_config


def test_repo_config_loads() -> None:
    config = load_config()
    assert config.backends == ["jev", "haiku"]
    assert config.labeler == "opus"


def test_unknown_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("random_seed: 1\nrandom_sed: 2\n")
    with pytest.raises(ValueError, match="random_sed"):
        load_config(path)


@pytest.mark.parametrize("db_path", [REPO_ROOT / "listings.db", REPO_ROOT / "data" / "x.db"])
def test_db_inside_repo_rejected(db_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the repo"):
        EvalConfig(db_path=str(db_path))


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        EvalConfig(backends=["gpt"])


def test_labeler_cannot_be_a_backend_under_test() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        EvalConfig(backends=["opus"])
    with pytest.raises(ValueError, match="Unknown labeler"):
        EvalConfig(labeler="haiku")
