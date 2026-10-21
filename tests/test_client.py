"""Tests for decision_model_eval.client and decision_model_eval.decision_log."""

from __future__ import annotations

import json
import threading
import urllib.error
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    HAIKU_CHOICE,
    JEV_CHOICE,
    JEV_NOUL,
    JEV_SCORE,
    SCIFACT_ITEM,
    STATE,
    VERDICT,
    FakeTransport,
    haiku_response,
    log_lines,
)

from decision_model_eval.client import DecisionError, decide
from decision_model_eval.decision_log import DecisionLog
from decision_model_eval.questions import Question, question_hash

pytestmark = pytest.mark.usefixtures("api_key")

NOUL = Question(type="noul", instructions="Is the abstract about the claim's topic?",
                criteria={"true": "It is about the topic", "false": "It is not"})
SCORE = Question(type="score", instructions="How strong is the evidence?",
                 criteria=["none", "weak", "moderate", "strong"])


def ask(log: DecisionLog, transport: FakeTransport, backend: str = "jev",
        question: Question = VERDICT, **options: Any) -> Any:
    sleeps: list[float] = options.pop("sleeps", [])
    return decide("scifact.verdict", options.pop("state", STATE), question,
                  backend=backend, item=SCIFACT_ITEM, log=log, transport=transport,
                  sleep=sleeps.append, **options)


def test_jev_choice_request_and_logged_line(log: DecisionLog) -> None:
    fake = FakeTransport([(200, JEV_CHOICE)])
    decision = ask(log, fake)
    call = fake.calls[0]
    assert call.url == "https://openrouter.ai/api/alpha/decisions"
    assert call.body == {"model": "typesafe/jev-1.14",
                         "provider": {"order": ["typesafe"], "allow_fallbacks": False},
                         "state": STATE, "questions": {"question": VERDICT.spec()}}
    assert call.headers["Authorization"] == "Bearer sk-or-test"
    assert (decision.answer, decision.confidence, decision.score) == ("supports", 0.97, None)
    assert decision.probabilities == {"supports": 0.98, "contradicts": 0.01, "nei": 0.01}

    header, line = log_lines(log.path)
    assert header == {"type": "batch", "format": 2, "batch_uuid": log.run_uuid,
                      "origin": "test", "run": {"purpose": "test"}}
    assert line["model"] == "typesafe/jev-1.14-20261014"  # Jev has no effort setting
    assert line["provider"] == "TypeSafe"
    assert (line["tokens_in"], line["tokens_out"], line["cost_usd"]) == (500, 60, 2.1e-05)
    assert (line["attempt"], line["error"], line["repeat_index"]) == (0, None, 0)
    assert line["run_uuid"] == log.run_uuid
    assert line["item"] == SCIFACT_ITEM
    assert line["input_payload"]["question_hash"] == question_hash(VERDICT)
    assert line["raw_response"] == JEV_CHOICE
    assert line == decision.record


def test_jev_noul_and_score(log: DecisionLog) -> None:
    noul = ask(log, FakeTransport([(200, JEV_NOUL)]), question=NOUL)
    assert noul.answer == "true"
    assert noul.probabilities == {"true": 0.97, "false": 0.03}
    assert noul.confidence is None  # Jev returns none for a noul
    score = ask(log, FakeTransport([(200, JEV_SCORE)]), question=SCORE)
    assert (score.answer, score.score, score.confidence) == ("2", 2.29, 0.46)
    assert score.record["output"]["score"] == 2.29


def test_haiku_request_pins_effort_schema_and_cache(log: DecisionLog) -> None:
    fake = FakeTransport([(200, HAIKU_CHOICE)])
    decision = ask(log, fake, backend="haiku")
    call = fake.calls[0]
    assert call.url == "https://openrouter.ai/api/v1/messages"
    assert call.headers["anthropic-version"] == "2023-06-01"
    body = call.body
    assert body["model"] == "anthropic/claude-haiku-5.5"
    assert body["provider"] == {"order": ["anthropic"], "allow_fallbacks": False}
    assert body["output_config"]["effort"] == "medium"
    schema = body["output_config"]["format"]["schema"]
    assert schema["properties"]["answer"]["enum"] == ["supports", "contradicts", "nei"]
    assert schema["properties"]["probabilities"]["required"] == VERDICT.options
    assert not {"temperature", "top_p", "top_k"} & body.keys()
    state_block, question_block = body["messages"][0]["content"]
    assert state_block["cache_control"] == {"type": "ephemeral"}
    assert json.dumps(STATE, indent=2) in state_block["text"]
    assert "- nei: The abstract gives no evidence either way" in question_block["text"]

    assert decision.answer == "supports"
    assert decision.probabilities == {"supports": 0.93, "contradicts": 0.03, "nei": 0.04}
    assert decision.confidence is None
    line = decision.record
    assert line["model"] == "anthropic/claude-haiku-5.5/effort=medium"
    assert line["tokens_in"] == 21 + 500 + 21  # cache writes and reads count as input
    assert line["tokens_out"] == 44


def test_haiku_noul_states_p_true(log: DecisionLog) -> None:
    fake = FakeTransport([(200, haiku_response('{"p_true": 0.8}'))])
    decision = ask(log, fake, backend="haiku", question=NOUL)
    assert fake.calls[0].body["output_config"]["format"]["schema"]["required"] == ["p_true"]
    assert (decision.answer, decision.probabilities) == ("true", {"true": 0.8, "false": 0.2})


def test_question_hash_is_shared_across_backends_and_template_hash_is_not(
    log: DecisionLog
) -> None:
    jev = ask(log, FakeTransport([(200, JEV_CHOICE)])).record
    haiku = ask(log, FakeTransport([(200, HAIKU_CHOICE)]), backend="haiku").record
    assert jev["input_payload"]["question_hash"] == haiku["input_payload"]["question_hash"]
    assert jev["prompt_template_hash"] != haiku["prompt_template_hash"]


def test_private_fields_are_stripped_before_sending_and_kept_in_the_log(
    log: DecisionLog
) -> None:
    state = {**STATE, "notes": "private"}
    fake = FakeTransport([(200, JEV_CHOICE)])
    decision = ask(log, fake, state=state, private_fields=["notes", "absent"])
    assert fake.calls[0].body["state"] == STATE
    payload = decision.record["input_payload"]
    assert payload["state"] == state
    assert (payload["private_fields"], payload["stripped"]) == (["absent", "notes"], ["notes"])


def test_retryable_status_is_retried_and_every_attempt_logged(log: DecisionLog) -> None:
    sleeps: list[float] = []
    fake = FakeTransport([(429, {"error": {"message": "rate limited"}}), (200, JEV_CHOICE)])
    decision = ask(log, fake, sleeps=sleeps)
    assert decision.answer == "supports"
    assert sleeps == [1.0]
    _, failed, succeeded = log_lines(log.path)
    assert (failed["attempt"], failed["error"], failed["output"]) == (0, "HTTP 429", None)
    assert failed["raw_response"] == {"error": {"message": "rate limited"}}
    assert (succeeded["attempt"], succeeded["error"]) == (1, None)


def test_network_errors_exhaust_the_retries(log: DecisionLog) -> None:
    sleeps: list[float] = []
    fake = FakeTransport([urllib.error.URLError("timed out") for _ in range(4)])
    with pytest.raises(DecisionError, match="attempt 4 of 4"):
        ask(log, fake, sleeps=sleeps)
    assert sleeps == [1.0, 2.0, 4.0]
    lines = log_lines(log.path)[1:]
    assert [line["attempt"] for line in lines] == [0, 1, 2, 3]
    assert all(line["error"].startswith("network error") for line in lines)


def test_client_errors_are_not_retried(log: DecisionLog) -> None:
    sleeps: list[float] = []
    fake = FakeTransport([(400, '{"error":{"message":"bad request"}}')])
    with pytest.raises(DecisionError, match="HTTP 400"):
        ask(log, fake, sleeps=sleeps)
    assert sleeps == []
    assert len(log_lines(log.path)) == 2


@pytest.mark.parametrize(("backend", "response", "message"), [
    ("jev", {**JEV_CHOICE, "model": "typesafe/jev-1.13-20260917"}, "pinned"),
    ("haiku", {**HAIKU_CHOICE, "provider": "Amazon Bedrock"}, "pinned 'anthropic'"),
    ("haiku", haiku_response('{"answer":"supports","probabilities":{"supports":1}}'),
     "probabilities must cover"),
    ("haiku", haiku_response('{"answer":"maybe","probabilities":'
                             '{"supports":0.5,"contradicts":0.3,"nei":0.2}}'), "not one of"),
    ("haiku", haiku_response('{"answer":"supp', stop_reason="max_tokens"), "stop_reason"),
    ("jev", "upstream gateway page", "unusable response"),
])
def test_unusable_responses_fail_and_are_logged(
    log: DecisionLog, backend: str, response: Any, message: str
) -> None:
    with pytest.raises(DecisionError, match=message):
        ask(log, FakeTransport([(200, response)]), backend=backend)
    line = log_lines(log.path)[-1]
    assert message.split()[0] in line["error"]
    assert line["raw_response"] == response


def test_a_paid_call_keeps_its_cost_when_the_pin_check_fails(log: DecisionLog) -> None:
    with pytest.raises(DecisionError):
        ask(log, FakeTransport([(200, {**JEV_CHOICE, "model": "typesafe/jev-latest"})]))
    assert log_lines(log.path)[-1]["cost_usd"] == 2.1e-05


def test_backend_without_pinned_effort_is_refused(log: DecisionLog) -> None:
    fake = FakeTransport([])
    with pytest.raises(ValueError, match="no pinned effort"):
        ask(log, fake, backend="opus")
    assert fake.calls == []


def test_missing_api_key(log: DecisionLog, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        ask(log, FakeTransport([]))


def test_log_refuses_an_existing_file(log: DecisionLog) -> None:
    with pytest.raises(FileExistsError):
        DecisionLog(log.path, origin="test", run={})


def test_log_lines_stay_whole_under_threads(tmp_path: Path) -> None:
    with DecisionLog(tmp_path / "threads.jsonl", origin="test", run={}) as shared:
        def write(n: int) -> None:
            for i in range(50):
                shared.write({"type": "decision", "writer": n, "i": i, "pad": "x" * 500})
        threads = [threading.Thread(target=write, args=(n,)) for n in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    lines = log_lines(shared.path)
    assert len(lines) == 1 + 8 * 50
