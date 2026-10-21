"""Tests for decision_model_eval.questions."""

from __future__ import annotations

from typing import Any

import pytest
from conftest import VERDICT, VERDICT_CRITERIA

from decision_model_eval.questions import Question, question_hash, template_hash


def test_options_per_type() -> None:
    assert VERDICT.options == ["supports", "contradicts", "nei"]
    noul = Question(type="noul", instructions="Remote?",
                    criteria={"true": "remote allowed", "false": "on site"})
    assert noul.options == ["true", "false"]
    score = Question(type="score", instructions="How strong?",
                     criteria=["none", "weak", "strong"])
    assert score.options == ["0", "1", "2"]


@pytest.mark.parametrize(("kind", "criteria", "message"), [
    ("noul", {"yes": "a", "no": "b"}, "exactly"),
    ("choice", {"only": "one"}, "2-255"),
    ("choice", ["a", "b"], "as an object"),
    ("score", ["one"], "2-10"),
    ("score", [str(i) for i in range(11)], "2-10"),
    ("score", {"0": "a", "1": "b"}, "2-10"),
    ("choice", {"a": "x", "b": " "}, "description"),
    ("verdict", {"a": "x", "b": "y"}, "unknown question type"),
])
def test_invalid_questions_are_rejected(kind: Any, criteria: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Question(type=kind, instructions="Q?", criteria=criteria)


def test_blank_instructions_are_rejected() -> None:
    with pytest.raises(ValueError, match="instructions"):
        Question(type="noul", instructions=" ", criteria={"true": "a", "false": "b"})


def test_spec_round_trips() -> None:
    assert Question.from_spec(VERDICT.spec()) == VERDICT
    with pytest.raises(ValueError, match="type, instructions, criteria"):
        Question.from_spec({"type": "noul", "instructions": "Q?"})


def test_question_hash_follows_the_wording_and_the_option_order() -> None:
    same = Question.from_spec(VERDICT.spec())
    reworded = Question(type="choice", instructions="Does the abstract back the claim?",
                        criteria=VERDICT_CRITERIA)
    reordered = Question(type="choice", instructions=VERDICT.instructions,
                         criteria=dict(reversed(VERDICT_CRITERIA.items())))
    assert question_hash(same) == question_hash(VERDICT)
    assert question_hash(reworded) != question_hash(VERDICT)
    assert question_hash(reordered) != question_hash(VERDICT)


def test_template_hash_adds_the_template() -> None:
    assert template_hash({"v": 1}, VERDICT) != template_hash({"v": 2}, VERDICT)
    assert template_hash({"v": 1}, VERDICT) != question_hash(VERDICT)
