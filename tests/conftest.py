"""Shared fixtures: a fresh store per test, a JSONL file writer, and a fake transport
for the decision client that replays canned API responses."""

from __future__ import annotations

import copy
import json
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from decision_model_eval.decision_log import DecisionLog
from decision_model_eval.questions import Question
from decision_model_eval.store import connect

WriteJsonl = Callable[..., Path]

# Canned responses, copied from live calls on 2026-10-20 (signatures and ids shortened).
JEV_CHOICE: dict[str, Any] = {
    "model": "typesafe/jev-1.14-20261014",
    "answers": {"question": {
        "type": "choice", "choice": "supports",
        "probabilities": {"supports": 0.98, "contradicts": 0.01, "nei": 0.01},
        "confidence": 0.97}},
    "usage": {"input_tokens": 500, "output_tokens": 60, "cost": 2.1e-05},
    "id": "gen-dec-1", "provider": "TypeSafe",
}
JEV_NOUL: dict[str, Any] = {
    **JEV_CHOICE, "answers": {"question": {"type": "noul", "noul": 0.97}},
}
JEV_SCORE: dict[str, Any] = {
    **JEV_CHOICE,
    "answers": {"question": {
        "type": "score", "score": 2.29,
        "legend": {"0": "none", "1": "weak", "2": "moderate", "3": "strong"},
        "probabilities": {"0": 0, "1": 0.12, "2": 0.46, "3": 0.42}, "confidence": 0.46}},
}

VERDICT_CRITERIA = {"supports": "The abstract gives evidence for the claim",
                    "contradicts": "The abstract gives evidence against the claim",
                    "nei": "The abstract gives no evidence either way"}
VERDICT = Question(type="choice", instructions="Does the abstract support or contradict "
                   "the claim?", criteria=VERDICT_CRITERIA)
STATE: dict[str, Any] = {"claim": "Daily aspirin reduces colorectal cancer incidence.",
                         "abstract": "Aspirin reduced 20-year colon cancer incidence."}
SCIFACT_ITEM: dict[str, Any] = {"dataset": "scifact", "claim_id": 5, "doc_id": 789}


def haiku_response(text: str, **overrides: Any) -> dict[str, Any]:
    response: dict[str, Any] = {
        "id": "msg_1", "type": "message", "role": "assistant",
        "model": "anthropic/claude-haiku-5.5",
        "content": [{"type": "thinking", "thinking": "", "signature": "CAQS"},
                    {"type": "text", "text": text, "citations": []}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 21, "output_tokens": 44,
                  "output_tokens_details": {"thinking_tokens": 0},
                  "cache_creation_input_tokens": 500, "cache_read_input_tokens": 21,
                  "cost": 7.62e-05},
        "provider": "Anthropic",
    }
    response.update(overrides)
    return response


HAIKU_CHOICE = haiku_response('{"answer":"supports","probabilities":'
                              '{"supports":0.93,"contradicts":0.03,"nei":0.04}}')


@dataclass
class Call:
    url: str
    body: dict[str, Any]
    headers: dict[str, str]


@dataclass
class FakeTransport:
    """Returns queued ``(status, body)`` pairs in order; a queued exception is raised."""

    replies: list[tuple[int, Any] | Exception]
    calls: list[Call] = field(default_factory=list)

    def __call__(self, url: str, body: dict[str, Any], headers: dict[str, str],
                 timeout: float) -> tuple[int, Any]:
        self.calls.append(Call(url, copy.deepcopy(body), dict(headers)))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply[0], copy.deepcopy(reply[1])


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")


@pytest.fixture
def log(tmp_path: Path) -> Iterator[DecisionLog]:
    with DecisionLog(tmp_path / "logs" / "run.jsonl", origin="test",
                     run={"purpose": "test"}) as decision_log:
        yield decision_log


def log_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


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
        "passed_filter": True, "company": "Acme Robotics",
        "title": "Software Engineer, Machine Learning Systems",
        "url": "https://acmerobotics.example/careers/software-engineer-ml-systems",
        "location": "Palo Alto, CA", "raw": {"id": 95451597, "weight": 0},
    }
    record.update(overrides)
    return record
