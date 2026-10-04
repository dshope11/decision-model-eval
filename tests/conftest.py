"""Shared fixtures: a fresh store per test and a JSONL file writer."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from decision_model_eval.store import connect

WriteJsonl = Callable[..., Path]


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection = connect(tmp_path / "store" / "listings.db")
    yield connection
    connection.close()


@pytest.fixture
def write_jsonl(tmp_path: Path) -> WriteJsonl:
    counter = iter(range(1_000_000))

    def write(*records: dict[str, Any]) -> Path:
        path = tmp_path / f"file-{next(counter)}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        return path

    return write


def run_header(**overrides: Any) -> dict[str, Any]:
    header: dict[str, Any] = {
        "type": "run", "format": 1, "run_uuid": "run-1", "source": "getro:103",
        "kind": "live", "started_at": "2026-10-03T20:00:00Z", "status": "ok",
        "complete": False,
        "queries": [{"query": "ml engineer", "params": {"limit": 60},
                     "results_reported": 2709, "results_fetched": 60}],
    }
    header.update(overrides)
    return header


def hit(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "type": "hit", "query": "ml engineer", "source_key": "95451597",
        "passed_filter": True, "company": "WindBorne Systems",
        "title": "Software Engineer, Machine Learning Systems",
        "url": "https://windbornesystems.com/careers/software-engineer-ml-systems",
        "location": "Palo Alto, CA", "raw": {"id": 95451597, "weight": 0},
    }
    record.update(overrides)
    return record
