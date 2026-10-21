"""The decision client: ``decide(point, state, question)`` against any registered backend.

One call answers one question about one state. Two adapters cover the registry:

- ``decisions`` (Jev): OpenRouter's typed-question endpoint, ``POST /api/alpha/decisions``,
  sent the state and the question spec unchanged. Jev returns a probability per option.
- ``messages`` (Haiku, Opus): OpenRouter's Anthropic-format endpoint, ``POST
  /api/v1/messages``, with the pinned effort in ``output_config.effort`` and a strict JSON
  schema in ``output_config.format``. The model states a probability per option, recorded
  as stated (not renormalized). The state is its own cached prompt block, so questions
  about one item reuse it when they run within the cache's five minutes.

``temperature`` and ``top_p`` are never sent: Anthropic's 5.5 models reject them, and
OpenRouter drops them silently, so a logged value would describe a setting that never
took effect.

Every attempt is one line in the decision log (``decision_log.py``):

    {"type": "decision", "item": object, "decision_point": str, "backend": str,
     "model": str, "provider": str | null, "prompt_template_hash": str,
     "input_payload": {"state": object, "question": object, "question_hash": str,
                       "private_fields": [str], "stripped": [str]},
     "output": {"answer": str, "probabilities": {option: float}, "score": float} | null,
     "confidence": float | null, "latency_ms": float, "tokens_in": int | null,
     "tokens_out": int | null, "cost_usd": float | null, "raw_response": any,
     "repeat_index": int, "attempt": int, "error": str | null, "run_uuid": str,
     "created_at": str}

``model`` is the backend's ``model_label``: the pinned model id, plus ``/effort=<level>``
where the backend has the setting. ``score`` is present only for Jev's score questions.
``tokens_in`` counts cached input too; the breakdown, thinking tokens included, is in
``raw_response``. The response must name the pinned model and provider, so a moved alias
or a fallback provider fails loudly instead of mixing into the data.

Two hashes make each line self-describing: ``question_hash`` identifies the wording and
is equal across backends; ``prompt_template_hash`` adds the backend's prompt template, so
it identifies what was actually sent.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any, Protocol

from decision_model_eval.backends import BACKENDS, BackendInfo
from decision_model_eval.decision_log import DecisionLog
from decision_model_eval.questions import Question, question_hash, template_hash
from decision_model_eval.store import utc_now

OPENROUTER = "https://openrouter.ai/api"
API_KEY_ENV = "OPENROUTER_API_KEY"
RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})
MAX_ATTEMPTS = 4
TIMEOUT_S = 120.0

# (url, JSON body, headers, timeout in seconds) -> (HTTP status, parsed body)
Transport = Callable[[str, dict[str, Any], dict[str, str], float], tuple[int, Any]]


class DecisionError(RuntimeError):
    """A decision that could not be made. Every attempt is already in the log."""


class ResponseError(ValueError):
    """A response that does not answer the question it was asked."""


@dataclass(frozen=True)
class Decision:
    """One backend's answer to one question."""

    answer: str
    probabilities: dict[str, float]
    confidence: float | None  # Jev's second axis on choice and score; None otherwise
    score: float | None  # Jev's fractional position on a score; None otherwise
    record: dict[str, Any]  # the line written to the log


@dataclass(frozen=True)
class _Parsed:
    output: dict[str, Any]
    confidence: float | None
    tokens_in: int | None
    tokens_out: int | None
    cost_usd: float | None
    model: Any
    provider: Any


def urllib_transport(url: str, body: dict[str, Any], headers: dict[str, str],
                     timeout: float) -> tuple[int, Any]:
    """POST ``body`` as JSON; return the status and the body (JSON, or text if it isn't).

    Network failures raise ``OSError``; HTTP errors return their status.
    """
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), _decode(response.read())
    except urllib.error.HTTPError as err:
        return err.code, _decode(err.read())


def _decode(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode("utf-8", "replace")


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _probability(value: Any, what: str) -> float:
    number = _float(value)
    if number is None or not 0 <= number <= 1:
        raise ResponseError(f"{what} is not a probability: {value!r}")
    return number


def _probabilities(raw: Any, question: Question) -> dict[str, float]:
    if not isinstance(raw, dict) or set(raw) != set(question.options):
        raise ResponseError(f"probabilities must cover exactly {question.options}: {raw!r}")
    return {k: _probability(raw[k], f"probability of {k!r}") for k in question.options}


def _answer(value: Any, question: Question) -> str:
    if value not in question.options:
        raise ResponseError(f"answer {value!r} is not one of {question.options}")
    return str(value)


def _noul(p_true: float) -> dict[str, Any]:
    return {"answer": "true" if p_true >= 0.5 else "false",
            "probabilities": {"true": p_true, "false": round(1 - p_true, 12)}}


def _pin(info: BackendInfo) -> dict[str, Any]:
    return {"order": [info.provider], "allow_fallbacks": False}


class _Adapter(Protocol):
    url: str
    headers: dict[str, str]

    def template(self, question: Question) -> dict[str, Any]: ...

    def request(self, info: BackendInfo, state: dict[str, Any],
                question: Question) -> dict[str, Any]: ...

    def parse(self, question: Question, response: Any) -> _Parsed: ...


class _Jev:
    """Jev's typed-question endpoint: the state and the spec go over unchanged."""

    url = f"{OPENROUTER}/alpha/decisions"
    headers: dict[str, str] = {}
    question_id = "question"

    def template(self, question: Question) -> dict[str, Any]:
        return {"request_version": 1, "question_id": self.question_id}

    def request(self, info: BackendInfo, state: dict[str, Any],
                question: Question) -> dict[str, Any]:
        return {"model": info.model, "provider": _pin(info), "state": state,
                "questions": {self.question_id: question.spec()}}

    def parse(self, question: Question, response: Any) -> _Parsed:
        answer = response["answers"][self.question_id]
        if answer.get("type") != question.type:
            raise ResponseError(f"answer type {answer.get('type')!r}, asked {question.type!r}")
        confidence = None
        if question.type == "noul":
            output = _noul(_probability(answer["noul"], "noul"))
        else:
            probabilities = _probabilities(answer["probabilities"], question)
            if question.type == "choice":
                output = {"answer": _answer(answer["choice"], question),
                          "probabilities": probabilities}
            else:
                score = _float(answer["score"])
                if score is None or not 0 <= score <= len(question.options) - 1:
                    raise ResponseError(f"score {answer['score']!r} is off the scale")
                output = {"answer": max(probabilities, key=probabilities.__getitem__),
                          "probabilities": probabilities, "score": score}
            confidence = _probability(answer["confidence"], "confidence")
        usage = response.get("usage") or {}
        return _Parsed(output=output, confidence=confidence,
                       tokens_in=_int(usage.get("input_tokens")),
                       tokens_out=_int(usage.get("output_tokens")),
                       cost_usd=_float(usage.get("cost")),
                       model=response.get("model"), provider=response.get("provider"))


FRAME = ("Answer one question about the state below, using only what the state says.\n\n"
         "State (JSON):\n{state}")
PROMPTS = {
    "choice": ("Question: {instructions}\n\nOptions:\n{criteria}\n\nReply with the option "
               'you choose as "answer" and your probability for every option as '
               '"probabilities", summing to 1.'),
    "score": ("Question: {instructions}\n\nLevels, lowest first:\n{criteria}\n\nReply with "
              'the level you choose as "answer" and your probability for every level as '
              '"probabilities", summing to 1.'),
    "noul": ("Question: {instructions}\n\n{criteria}\n\nReply with your probability that "
             'the answer is true as "p_true", between 0 and 1.'),
}
MAX_TOKENS = 16000  # a cap on thinking plus answer, not a target


class _Messages:
    """Anthropic-format messages with a strict schema; the model states its probabilities."""

    url = f"{OPENROUTER}/v1/messages"
    headers = {"anthropic-version": "2023-06-01"}

    def template(self, question: Question) -> dict[str, Any]:
        return {"request_version": 1, "frame": FRAME, "prompt": PROMPTS[question.type],
                "schema": self._schema(question), "max_tokens": MAX_TOKENS}

    @staticmethod
    def _schema(question: Question) -> dict[str, Any]:
        if question.type == "noul":
            return {"type": "object", "properties": {"p_true": {"type": "number"}},
                    "required": ["p_true"], "additionalProperties": False}
        options = question.options
        return {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": options},
                "probabilities": {
                    "type": "object",
                    "properties": {o: {"type": "number"} for o in options},
                    "required": options, "additionalProperties": False,
                },
            },
            "required": ["answer", "probabilities"], "additionalProperties": False,
        }

    @staticmethod
    def _prompt(question: Question) -> str:
        if isinstance(question.criteria, list):
            pairs = [(str(i), d) for i, d in enumerate(question.criteria)]
        else:
            pairs = list(question.criteria.items())
        criteria = "\n".join(f"- {k}: {v}" for k, v in pairs)
        return PROMPTS[question.type].format(instructions=question.instructions,
                                             criteria=criteria)

    def request(self, info: BackendInfo, state: dict[str, Any],
                question: Question) -> dict[str, Any]:
        frame = FRAME.format(state=json.dumps(state, indent=2, ensure_ascii=False))
        return {
            "model": info.model, "max_tokens": MAX_TOKENS, "provider": _pin(info),
            "output_config": {"effort": info.effort,
                              "format": {"type": "json_schema",
                                         "schema": self._schema(question)}},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": frame, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": self._prompt(question)},
            ]}],
        }

    def parse(self, question: Question, response: Any) -> _Parsed:
        if response.get("stop_reason") != "end_turn":
            raise ResponseError(f"stop_reason {response.get('stop_reason')!r}")
        text = "".join(b.get("text", "") for b in response["content"] if b.get("type") == "text")
        data = json.loads(text)
        if question.type == "noul":
            output = _noul(_probability(data["p_true"], "p_true"))
        else:
            output = {"answer": _answer(data["answer"], question),
                      "probabilities": _probabilities(data["probabilities"], question)}
        usage = response.get("usage") or {}
        input_parts = [_int(usage.get(k)) for k in (
            "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")]
        tokens_in = None if input_parts[0] is None else sum(p or 0 for p in input_parts)
        return _Parsed(output=output, confidence=None, tokens_in=tokens_in,
                       tokens_out=_int(usage.get("output_tokens")),
                       cost_usd=_float(usage.get("cost")),
                       model=response.get("model"), provider=response.get("provider"))


_ADAPTERS: dict[str, _Adapter] = {"decisions": _Jev(), "messages": _Messages()}


def _same_provider(reported: Any, pinned: str) -> bool:
    def norm(name: str) -> str:
        return "".join(c for c in name.lower() if c.isalnum())
    return isinstance(reported, str) and norm(reported) == norm(pinned)


def _api_key() -> str:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"{API_KEY_ENV} is not set")
    return key


def decide(
    point: str,
    state: dict[str, Any],
    question: Question,
    *,
    backend: str,
    item: dict[str, Any],
    log: DecisionLog,
    repeat_index: int = 0,
    private_fields: Collection[str] = (),
    transport: Transport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int = MAX_ATTEMPTS,
    timeout: float = TIMEOUT_S,
) -> Decision:
    """Ask ``backend`` one question about ``state``, log every attempt, return the answer.

    ``point`` names the decision point (``"D1"``, ``"D2.G3"``, ``"scifact.verdict"``).
    ``item`` identifies what the state describes: ``{"listing_id": int}`` (or ``source`` +
    ``source_key``) for a listing, which ``dme record`` can store, or any other object,
    e.g. ``{"dataset": "scifact", "claim_id": 5, "doc_id": 789}``. ``private_fields`` are
    state keys that must not leave the machine: they are removed before a remote backend
    sees the state, and kept in the logged input so a replay is exact.

    HTTP 408 / 429 / 5xx and network errors are retried with backoff (1, 2, 4 s), each
    attempt logged. Raises ``DecisionError`` when retries run out or a response is unusable.
    """
    info = BACKENDS[backend]
    if info.api == "messages" and info.effort is None:
        raise ValueError(f"backend {backend!r} has no pinned effort; set it in backends.py")
    if not point.strip():
        raise ValueError("point must be a non-empty string")
    if not isinstance(state, dict):
        raise TypeError("state must be a JSON object")
    if not isinstance(item, dict) or not item:
        raise ValueError("item must be a non-empty JSON object")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    adapter = _ADAPTERS[info.api]
    declared = sorted(set(private_fields))
    stripped = [k for k in declared if k in state] if info.remote else []
    body = adapter.request(info, {k: v for k, v in state.items() if k not in stripped},
                           question)
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json",
               **adapter.headers}
    base: dict[str, Any] = {
        "type": "decision", "item": item, "decision_point": point, "backend": backend,
        "model": info.model_label,
        "prompt_template_hash": template_hash(adapter.template(question), question),
        "input_payload": {"state": state, "question": question.spec(),
                          "question_hash": question_hash(question),
                          "private_fields": declared, "stripped": stripped},
        "repeat_index": repeat_index, "run_uuid": log.run_uuid,
    }
    send = transport or urllib_transport

    for attempt in range(max_attempts):
        started = time.perf_counter()
        status: int | None = None
        response: Any = None
        network_error = None
        try:
            status, response = send(adapter.url, body, headers, timeout)
        except OSError as err:
            network_error = f"network error: {err}"
        line: dict[str, Any] = {
            **base, "provider": None, "output": None, "confidence": None,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "tokens_in": None, "tokens_out": None, "cost_usd": None,
            "raw_response": response, "attempt": attempt, "error": None,
            "created_at": utc_now(),
        }
        where = f"{backend} {point} {item} (attempt {attempt + 1} of {max_attempts})"

        if network_error is None and status == 200:
            try:
                parsed = adapter.parse(question, response)
            except (ResponseError, KeyError, TypeError, ValueError, AttributeError) as err:
                raise _failure(log, line, where,
                               f"unusable response: {type(err).__name__}: {err}") from err
            line.update(provider=parsed.provider, output=parsed.output,
                        confidence=parsed.confidence, tokens_in=parsed.tokens_in,
                        tokens_out=parsed.tokens_out, cost_usd=parsed.cost_usd)
            if parsed.model != info.resolved_model:
                raise _failure(log, line, where, f"response model {parsed.model!r}, "
                                                 f"pinned {info.resolved_model!r}")
            if not _same_provider(parsed.provider, info.provider):
                raise _failure(log, line, where, f"response provider {parsed.provider!r}, "
                                                 f"pinned {info.provider!r}")
            log.write(line)
            return Decision(answer=parsed.output["answer"],
                            probabilities=parsed.output["probabilities"],
                            confidence=parsed.confidence, score=parsed.output.get("score"),
                            record=line)

        error = network_error or f"HTTP {status}"
        if (status is not None and status not in RETRY_STATUS) or attempt == max_attempts - 1:
            raise _failure(log, line, where, error)
        line["error"] = error
        log.write(line)
        sleep(2.0 ** attempt)
    raise AssertionError("unreachable")


def _failure(log: DecisionLog, line: dict[str, Any], where: str, error: str) -> DecisionError:
    """Log a failed attempt and return the error to raise."""
    line["error"] = error
    log.write(line)
    return DecisionError(f"{where}: {error}")
