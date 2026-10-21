"""Typed questions, in Jev's wire format, and the hashes that identify them.

A question is one of three types (the names are Jev's):

- ``noul``: yes / no. ``criteria`` is ``{"true": str, "false": str}``.
- ``choice``: one of 2-255 options. ``criteria`` maps option id -> description, in
  the order the options are presented.
- ``score``: a position on 2-10 ordered levels. ``criteria`` lists the level
  descriptions, lowest first; the level ids are ``"0"`` .. ``"N-1"``, as Jev returns them.

Every backend is sent the same spec, so ``question_hash`` - a hash of the spec alone - is
equal across backends and identifies the wording: analysis pairs backends on it and
refuses to pool rows asked in different words. ``template_hash`` adds a backend's prompt
template, so it identifies what was actually sent.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

from decision_model_eval.jsonl import canonical_json

QuestionType = Literal["noul", "choice", "score"]
NOUL_OPTIONS = ("true", "false")
MAX_CHOICE_OPTIONS = 255
SCORE_LEVELS = range(2, 11)


@dataclass(frozen=True)
class Question:
    """One typed question. Validated on construction."""

    type: QuestionType
    instructions: str
    criteria: dict[str, str] | list[str]

    def __post_init__(self) -> None:
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError("instructions must be a non-empty string")
        if self.type == "score":
            if not isinstance(self.criteria, list) or len(self.criteria) not in SCORE_LEVELS:
                raise ValueError("a score needs a list of 2-10 level descriptions")
            descriptions = list(self.criteria)
        elif self.type in ("noul", "choice"):
            if not isinstance(self.criteria, dict):
                raise ValueError(f"a {self.type} needs criteria as an object")
            if self.type == "noul" and set(self.criteria) != set(NOUL_OPTIONS):
                raise ValueError('a noul needs exactly the criteria "true" and "false"')
            if self.type == "choice" and not 2 <= len(self.criteria) <= MAX_CHOICE_OPTIONS:
                raise ValueError(f"a choice needs 2-{MAX_CHOICE_OPTIONS} options")
            if not all(isinstance(k, str) and k.strip() for k in self.criteria):
                raise ValueError("option ids must be non-empty strings")
            descriptions = list(self.criteria.values())
        else:
            raise ValueError(f"unknown question type {self.type!r}")
        if not all(isinstance(d, str) and d.strip() for d in descriptions):
            raise ValueError("every criterion needs a non-empty description")

    @property
    def options(self) -> list[str]:
        """The answer ids a backend returns probabilities for."""
        if self.type == "noul":
            return list(NOUL_OPTIONS)
        if isinstance(self.criteria, list):
            return [str(i) for i in range(len(self.criteria))]
        return list(self.criteria)

    def spec(self) -> dict[str, Any]:
        """The question in Jev's wire format, as sent and as logged."""
        criteria = dict(self.criteria) if isinstance(self.criteria, dict) else list(self.criteria)
        return {"type": self.type, "instructions": self.instructions, "criteria": criteria}

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> Question:
        """Rebuild a question from its logged spec."""
        if set(spec) != {"type", "instructions", "criteria"}:
            raise ValueError(f"a question spec has type, instructions, criteria: {sorted(spec)}")
        return cls(type=spec["type"], instructions=spec["instructions"],
                   criteria=spec["criteria"])


def content_hash(value: Any) -> str:
    """First 16 hex digits of the SHA-256 of ``value``'s canonical JSON."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:16]


def _ordered(question: Question) -> dict[str, Any]:
    # canonical_json sorts object keys, which would make two presentation orders of the
    # same options hash equal; a list of pairs keeps the order a backend sees.
    spec = question.spec()
    if isinstance(question.criteria, dict):
        spec["criteria"] = [[k, v] for k, v in question.criteria.items()]
    return spec


def question_hash(question: Question) -> str:
    """Identifies the wording: equal for every backend asked the same question."""
    return content_hash(_ordered(question))


def template_hash(template: Any, question: Question) -> str:
    """Identifies what one backend was sent: its prompt template plus the wording."""
    return content_hash({"template": template, "question": _ordered(question)})
