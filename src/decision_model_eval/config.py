"""Configuration dataclass and YAML loader.

``EvalConfig`` validates its fields in ``__post_init__`` so bad config is caught
at load time, not partway through a run of paid API calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from decision_model_eval.backends import BACKENDS

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class EvalConfig:
    """All knobs for the evaluation, loaded from ``config.yaml`` via ``load_config()``."""

    # The listings database holds scraped postings and private labels. It lives
    # outside the repo and is never committed.
    db_path: str = "~/.local/share/decision-model-eval/listings.db"
    random_seed: int = 42
    backends: list[str] = field(default_factory=lambda: ["jev", "haiku"])
    labeler: str = "opus"

    def __post_init__(self) -> None:
        if self.resolved_db_path().is_relative_to(REPO_ROOT):
            raise ValueError(f"db_path must be outside the repo ({REPO_ROOT}), got {self.db_path}")

        if self.random_seed < 0:
            raise ValueError(f"random_seed must be non-negative, got {self.random_seed}")

        if not self.backends:
            raise ValueError("backends must name at least one backend")
        for key in self.backends:
            if key not in BACKENDS or BACKENDS[key].role != "backend":
                raise ValueError(f"Unknown backend {key!r}; registered: {sorted(BACKENDS)}")
        if self.labeler not in BACKENDS or BACKENDS[self.labeler].role != "labeler":
            raise ValueError(f"Unknown labeler {self.labeler!r}")

    def resolved_db_path(self) -> Path:
        """``db_path`` with ``~`` expanded, made absolute."""
        return Path(self.db_path).expanduser().resolve()


def load_config(path: str | Path = REPO_ROOT / "config.yaml") -> EvalConfig:
    """Load and validate an ``EvalConfig`` from a YAML file.

    Unknown keys raise a ``ValueError`` - this catches typos in config.yaml at
    load time rather than silently ignoring them.
    """
    path = Path(path)
    with path.open("r") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    valid_fields = {f.name for f in fields(EvalConfig)}
    unknown = set(raw) - valid_fields
    if unknown:
        raise ValueError(
            f"Unknown config keys in {path}: {sorted(unknown)}. Valid keys: {sorted(valid_fields)}"
        )
    return EvalConfig(**raw)
