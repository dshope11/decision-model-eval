"""Replay a decision log: ask its logged inputs again, against any backend.

Every successful line of a log carries its full input - the state before any field was
stripped, the question spec and the declared private fields - so a log replays exactly:
against the same backend for a run-to-run noise check, or against another (a new model
version, a future local model) for a comparison on identical inputs. Private fields are
stripped again for a remote backend. Failed lines are not replayed; their inputs appear
on a successful line too, or the item never got an answer to compare against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decision_model_eval import jsonl
from decision_model_eval.client import Decision, DecisionError, decide
from decision_model_eval.decision_log import LOG_FORMAT, DecisionLog
from decision_model_eval.jsonl import FormatError
from decision_model_eval.questions import Question, question_hash


@dataclass(frozen=True)
class LoggedInput:
    """One distinct input of a decision log."""

    point: str
    item: dict[str, Any]
    state: dict[str, Any]
    question: Question
    private_fields: tuple[str, ...]


def logged_inputs(path: str | Path) -> list[LoggedInput]:
    """The distinct inputs of a log's successful lines, in first-seen order."""
    records = jsonl.read_records(path)
    header_line, header = records[0]
    if header.get("type") != "batch" or header.get("format") != LOG_FORMAT:
        raise FormatError(f"line {header_line}: not a decision log (format {LOG_FORMAT})")
    seen: set[str] = set()
    inputs = []
    for line, record in records[1:]:
        if record.get("type") != "decision" or record.get("error") is not None:
            continue
        payload = record["input_payload"]
        try:
            question = Question.from_spec(payload["question"])
        except (KeyError, TypeError, ValueError) as err:
            raise FormatError(f"line {line}: bad question spec ({err})") from err
        if question_hash(question) != payload["question_hash"]:
            raise FormatError(f"line {line}: question_hash does not match the question spec")
        key = jsonl.canonical_json([record["decision_point"], record["item"],
                                    payload["question_hash"], payload["state"]])
        if key in seen:
            continue
        seen.add(key)
        inputs.append(LoggedInput(point=record["decision_point"], item=record["item"],
                                  state=payload["state"], question=question,
                                  private_fields=tuple(payload["private_fields"])))
    return inputs


def replay(path: str | Path, *, backend: str, log: DecisionLog, repeats: int = 1,
           **decide_options: Any) -> tuple[list[Decision], int]:
    """Ask every distinct input of the log at ``path`` again, ``repeats`` times each.

    Returns the decisions and the number that failed (those are in ``log`` too).
    ``decide_options`` pass through to ``decide`` (a test transport, for one).
    """
    decisions, failed = [], 0
    for logged in logged_inputs(path):
        for repeat in range(repeats):
            try:
                decisions.append(decide(
                    logged.point, logged.state, logged.question, backend=backend,
                    item=logged.item, log=log, repeat_index=repeat,
                    private_fields=logged.private_fields, **decide_options))
            except DecisionError:
                failed += 1
    return decisions, failed
