"""Tests for decision_model_eval.replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import (
    HAIKU_CHOICE,
    JEV_CHOICE,
    SCIFACT_ITEM,
    STATE,
    VERDICT,
    FakeTransport,
    WriteJsonl,
    log_lines,
)

from decision_model_eval.client import decide
from decision_model_eval.decision_log import DecisionLog
from decision_model_eval.jsonl import FormatError
from decision_model_eval.replay import logged_inputs, replay

pytestmark = pytest.mark.usefixtures("api_key")

OTHER_ITEM = {"dataset": "scifact", "claim_id": 6, "doc_id": 790}
PRIVATE_STATE = {**STATE, "notes": "private"}


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A Jev log: item 1 asked twice, item 2 once after a rate-limited attempt."""
    fake = FakeTransport([(200, JEV_CHOICE), (200, JEV_CHOICE),
                          (429, {"error": "rate limited"}), (200, JEV_CHOICE)])
    with DecisionLog(tmp_path / "source.jsonl", origin="test", run={}) as log:
        for repeat in range(2):
            decide("scifact.verdict", PRIVATE_STATE, VERDICT, backend="jev",
                   item=SCIFACT_ITEM, log=log, repeat_index=repeat,
                   private_fields=["notes"], transport=fake, sleep=lambda _: None)
        decide("scifact.verdict", STATE, VERDICT, backend="jev", item=OTHER_ITEM, log=log,
               transport=fake, sleep=lambda _: None)
    return log.path


def test_logged_inputs_are_distinct_and_skip_failed_attempts(source: Path) -> None:
    inputs = logged_inputs(source)
    assert [i.item for i in inputs] == [SCIFACT_ITEM, OTHER_ITEM]
    first = inputs[0]
    assert (first.point, first.state, first.question) == ("scifact.verdict", PRIVATE_STATE,
                                                          VERDICT)
    assert first.private_fields == ("notes",)


def test_replay_against_another_backend(source: Path, tmp_path: Path) -> None:
    fake = FakeTransport([(200, HAIKU_CHOICE)] * 4)
    with DecisionLog(tmp_path / "replay.jsonl", origin="test", run={}) as log:
        decisions, failed = replay(source, backend="haiku", log=log, repeats=2,
                                   transport=fake)
    assert (len(decisions), failed) == (4, 0)
    assert "private" not in json.dumps(fake.calls[0].body)  # stripped again
    source_line = log_lines(source)[1]
    replayed = log_lines(log.path)[1:]
    assert [line["repeat_index"] for line in replayed] == [0, 1, 0, 1]
    assert {line["backend"] for line in replayed} == {"haiku"}
    assert replayed[0]["input_payload"] == source_line["input_payload"]


def test_replay_counts_failures(source: Path, tmp_path: Path) -> None:
    fake = FakeTransport([(200, HAIKU_CHOICE), (400, "bad request")])
    with DecisionLog(tmp_path / "replay.jsonl", origin="test", run={}) as log:
        decisions, failed = replay(source, backend="haiku", log=log, transport=fake)
    assert (len(decisions), failed) == (1, 1)


def test_only_decision_logs_replay(write_jsonl: WriteJsonl) -> None:
    with pytest.raises(FormatError, match="not a decision log"):
        logged_inputs(write_jsonl({"type": "batch", "format": 1, "batch_uuid": "b"}))


def test_a_spec_that_does_not_match_its_hash_is_refused(source: Path,
                                                        write_jsonl: WriteJsonl) -> None:
    header, line, *_ = log_lines(source)
    line["input_payload"]["question"]["instructions"] = "Edited after the run?"
    with pytest.raises(FormatError, match="question_hash"):
        logged_inputs(write_jsonl(header, line))
