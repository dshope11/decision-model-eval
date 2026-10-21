"""Record decisions, labels and pipeline events from an annotations JSONL file.

Format 1. **Line 1 is the batch header**, whose UUID makes re-recording a no-op:

    {"type": "batch", "format": 1, "batch_uuid": str, optional: "origin"}

**Every later line is one annotation of an existing listing**, identified by
``"listing": {"source": str, "source_key": str}`` or by the store id that
``dme surfaced`` prints, ``"listing": {"listing_id": int}``. Decisions and labels attach to
the listing's latest snapshot (the content the decider saw) unless an explicit
``snapshot_id`` is given.

    {"type": "decision", "listing": ..., "decision_point": str, "backend": str,
     "output": object, optional: "snapshot_id", "model", "provider",
     "prompt_template_hash", "input_payload", "confidence", "latency_ms",
     "tokens_in", "tokens_out", "cost_usd", "raw_response", "repeat_index",
     "created_at"}
    {"type": "label", "listing": ..., "decision_point": str, "labeler": str,
     "value": any JSON, "saw_model_output": bool, optional: "snapshot_id",
     "sample", "rubric_version", "labeled_at", "note"}
    {"type": "event", "listing": ..., "event": str, "at": timestamp,
     optional: "channel", "drop_reason", "evidence", "note"}

**Inline decisions carry their effort level.** For backend ``claude-inline`` (the
judgments Claude makes inside a workflow session) the reasoning-effort setting is
part of the backend's identity, like a pinned model version: ``model`` must be
``"<model id>/effort=<level>"``, e.g. ``"claude-opus-5-5/effort=high"``. Without
it, a run at a different effort level would silently mix into the baseline.

**Inline decisions also name the listing view they were made on**, as
``"input_payload": {"listing_view": "v2"}`` - the version ``dme surfaced`` prints.
Blind labels record theirs as the rubric version, and agreement is only measured
between judgments made on the same information. Batches recorded before the
requirement (the 2026-10-03 run, view v1) carry none.

**Format 2 is the decision log** that ``decide()`` writes (``decision_log.py``,
``client.py``). The header may add a ``"run"`` object. Decision lines name what they
decided as ``"item"`` instead of ``"listing"`` and add ``"run_uuid"``, ``"attempt"`` and
``"error"``; ``raw_response`` may be any JSON. Only listing items are recorded - an item
that is not a listing (a SciFact pair) is an error, since such a log stays as JSONL - and
failed attempts (``error`` set) are skipped and counted. Labels and events are written as
in format 1.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decision_model_eval import jsonl
from decision_model_eval.jsonl import FormatError
from decision_model_eval.store import utc_now

_DECISION_REQUIRED = {"type", "listing", "decision_point", "backend", "output"}
_DECISION_OPTIONAL = {
    "snapshot_id", "model", "provider", "prompt_template_hash", "input_payload",
    "confidence", "latency_ms", "tokens_in", "tokens_out", "cost_usd", "raw_response",
    "repeat_index", "created_at",
}
_DECISION2_REQUIRED = _DECISION_REQUIRED - {"listing"} | {"item"}
_DECISION2_OPTIONAL = _DECISION_OPTIONAL | {"run_uuid", "attempt", "error"}
FORMATS = (1, 2)
INLINE_BACKEND = "claude-inline"
_INLINE_MODEL = re.compile(r"^[\w.-]+/effort=(low|medium|high|xhigh|max)$")
_LISTING_VIEW = re.compile(r"^v\d+$")
_LABEL_REQUIRED = {"type", "listing", "decision_point", "labeler", "value", "saw_model_output"}
_LABEL_OPTIONAL = {"snapshot_id", "sample", "rubric_version", "labeled_at", "note"}
_EVENT_REQUIRED = {"type", "listing", "event", "at"}
_EVENT_OPTIONAL = {"channel", "drop_reason", "evidence", "note"}


@dataclass(frozen=True)
class RecordResult:
    batch_uuid: str
    already_recorded: bool
    decisions: int = 0
    labels: int = 0
    events: int = 0
    skipped: int = 0  # failed attempts in a decision log, not recorded


def record_file(conn: sqlite3.Connection, path: str | Path) -> RecordResult:
    """Validate and record one annotations file in a single transaction."""
    records = jsonl.read_records(path)
    header_line, header = records[0]
    if header.get("type") != "batch":
        raise FormatError(f"line {header_line}: the first record must be the batch header")
    fmt = header.get("format")
    if fmt not in FORMATS:
        raise FormatError(f"line {header_line}: unsupported format {fmt!r}")
    jsonl.check_keys(header, header_line, {"type", "format", "batch_uuid"},
                     {"origin"} if fmt == 1 else {"origin", "run"})
    batch_uuid = jsonl.required_text(header["batch_uuid"], header_line, "batch_uuid")
    if conn.execute(
        "SELECT 1 FROM record_batches WHERE batch_uuid = ?", (batch_uuid,)
    ).fetchone():
        return RecordResult(batch_uuid=batch_uuid, already_recorded=True)

    counts = {"decision": 0, "label": 0, "event": 0}
    skipped = 0
    conn.execute("BEGIN")
    try:
        batch_id = conn.execute(
            "INSERT INTO record_batches (batch_uuid, origin, recorded_at) VALUES (?, ?, ?)",
            (batch_uuid, jsonl.text(header.get("origin"), header_line, "origin"), utc_now()),
        ).lastrowid
        for line, record in records[1:]:
            kind = record.get("type")
            if kind == "decision" and fmt == 2:
                if not _logged_decision(conn, batch_id, record, line):
                    skipped += 1
                    continue
            elif kind == "decision":
                _decision(conn, batch_id, record, line)
            elif kind == "label":
                _label(conn, batch_id, record, line)
            elif kind == "event":
                _event(conn, batch_id, record, line)
            else:
                raise FormatError(f"line {line}: unknown record type {kind!r}")
            counts[kind] += 1
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return RecordResult(
        batch_uuid=batch_uuid, already_recorded=False, decisions=counts["decision"],
        labels=counts["label"], events=counts["event"], skipped=skipped,
    )


def _listing_id(conn: sqlite3.Connection, record: dict[str, Any], line: int,
                key: str = "listing") -> int:
    ref = record[key]
    if not isinstance(ref, dict):
        raise FormatError(f"line {line}: {key} must be an object")
    if "listing_id" in ref:
        jsonl.check_keys(ref, line, {"listing_id"}, set())
        listing_id = jsonl.integer(ref["listing_id"], line, "listing.listing_id")
        row = conn.execute("SELECT id FROM listings WHERE id = ?", (listing_id,)).fetchone()
        described = f"listing_id {listing_id}"
    else:
        jsonl.check_keys(ref, line, {"source", "source_key"}, set())
        row = conn.execute(
            "SELECT id FROM listings WHERE source = ? AND source_key = ?",
            (ref["source"], ref["source_key"]),
        ).fetchone()
        described = f"({ref['source']!r}, {ref['source_key']!r})"
    if row is None:
        raise FormatError(f"line {line}: no listing {described}; ingest its fetch run first")
    return int(row["id"])


def _snapshot_id(conn: sqlite3.Connection, record: dict[str, Any], line: int,
                 key: str = "listing") -> int:
    listing_id = _listing_id(conn, record, line, key)
    explicit = jsonl.integer(record.get("snapshot_id"), line, "snapshot_id")
    if explicit is not None:
        row = conn.execute(
            "SELECT id FROM snapshots WHERE id = ? AND listing_id = ?", (explicit, listing_id)
        ).fetchone()
        if row is None:
            raise FormatError(f"line {line}: snapshot {explicit} does not belong to the listing")
        return explicit
    row = conn.execute(
        "SELECT id FROM latest_snapshots WHERE listing_id = ?", (listing_id,)
    ).fetchone()
    if row is None:
        raise FormatError(f"line {line}: the listing has no snapshot")
    return int(row["id"])


def _optional_json(value: Any) -> str | None:
    return None if value is None else jsonl.canonical_json(value)


def _decision(conn: sqlite3.Connection, batch_id: int | None, rec: dict[str, Any],
              line: int) -> None:
    jsonl.check_keys(rec, line, _DECISION_REQUIRED, _DECISION_OPTIONAL)
    _insert_decision(conn, batch_id, rec, line, "listing",
                     jsonl.text(rec.get("raw_response"), line, "raw_response"))


def _logged_decision(conn: sqlite3.Connection, batch_id: int | None, rec: dict[str, Any],
                     line: int) -> bool:
    """Record one decision-log line; False for a failed attempt, which is skipped."""
    jsonl.check_keys(rec, line, _DECISION2_REQUIRED, _DECISION2_OPTIONAL)
    if rec.get("error") is not None:
        return False
    item = rec["item"]
    if not isinstance(item, dict) or set(item) not in ({"listing_id"}, {"source", "source_key"}):
        raise FormatError(f"line {line}: item {item!r} is not a listing; only listing "
                          "decisions are recorded, so this log stays as JSONL")
    raw = rec.get("raw_response")
    _insert_decision(conn, batch_id, rec, line, "item",
                     raw if raw is None or isinstance(raw, str) else jsonl.canonical_json(raw))
    return True


def _insert_decision(conn: sqlite3.Connection, batch_id: int | None, rec: dict[str, Any],
                     line: int, key: str, raw_response: str | None) -> None:
    if rec["backend"] == INLINE_BACKEND and not _INLINE_MODEL.match(str(rec.get("model"))):
        raise FormatError(
            f"line {line}: {INLINE_BACKEND} decisions need model "
            f"'<model id>/effort=<level>', got {rec.get('model')!r}"
        )
    if rec["backend"] == INLINE_BACKEND:
        payload = rec.get("input_payload")
        view = payload.get("listing_view") if isinstance(payload, dict) else None
        if not _LISTING_VIEW.match(str(view)):
            raise FormatError(
                f"line {line}: {INLINE_BACKEND} decisions need input_payload "
                f"{{\"listing_view\": \"v<N>\"}}, got {payload!r}"
            )
    created_at = (
        jsonl.timestamp(rec["created_at"], line, "created_at")
        if rec.get("created_at") is not None else utc_now()
    )
    conn.execute(
        "INSERT INTO decisions (batch_id, snapshot_id, decision_point, backend, model, "
        "provider, prompt_template_hash, input_payload, output, confidence, latency_ms, "
        "tokens_in, tokens_out, cost_usd, raw_response, repeat_index, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (batch_id, _snapshot_id(conn, rec, line, key),
         jsonl.required_text(rec["decision_point"], line, "decision_point"),
         jsonl.required_text(rec["backend"], line, "backend"),
         jsonl.text(rec.get("model"), line, "model"),
         jsonl.text(rec.get("provider"), line, "provider"),
         jsonl.text(rec.get("prompt_template_hash"), line, "prompt_template_hash"),
         _optional_json(rec.get("input_payload")), jsonl.canonical_json(rec["output"]),
         jsonl.number(rec.get("confidence"), line, "confidence"),
         jsonl.number(rec.get("latency_ms"), line, "latency_ms"),
         jsonl.integer(rec.get("tokens_in"), line, "tokens_in"),
         jsonl.integer(rec.get("tokens_out"), line, "tokens_out"),
         jsonl.number(rec.get("cost_usd"), line, "cost_usd"),
         raw_response,
         jsonl.integer(rec.get("repeat_index"), line, "repeat_index") or 0,
         created_at),
    )


def _label(conn: sqlite3.Connection, batch_id: int | None, rec: dict[str, Any],
           line: int) -> None:
    jsonl.check_keys(rec, line, _LABEL_REQUIRED, _LABEL_OPTIONAL)
    labeled_at = (
        jsonl.timestamp(rec["labeled_at"], line, "labeled_at")
        if rec.get("labeled_at") is not None else utc_now()
    )
    conn.execute(
        "INSERT INTO labels (batch_id, snapshot_id, decision_point, labeler, value, sample, "
        "saw_model_output, rubric_version, labeled_at, note) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (batch_id, _snapshot_id(conn, rec, line),
         jsonl.required_text(rec["decision_point"], line, "decision_point"),
         jsonl.required_text(rec["labeler"], line, "labeler"),
         jsonl.canonical_json(rec["value"]),
         jsonl.text(rec.get("sample"), line, "sample"),
         jsonl.boolean(rec["saw_model_output"], line, "saw_model_output"),
         jsonl.text(rec.get("rubric_version"), line, "rubric_version"),
         labeled_at, jsonl.text(rec.get("note"), line, "note")),
    )


def _event(conn: sqlite3.Connection, batch_id: int | None, rec: dict[str, Any],
           line: int) -> None:
    jsonl.check_keys(rec, line, _EVENT_REQUIRED, _EVENT_OPTIONAL)
    try:
        conn.execute(
            "INSERT INTO pipeline_events (batch_id, listing_id, event, at, channel, "
            "drop_reason, evidence, note, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (batch_id, _listing_id(conn, rec, line),
             jsonl.required_text(rec["event"], line, "event"),
             jsonl.date_or_timestamp(rec["at"], line, "at"),
             jsonl.text(rec.get("channel"), line, "channel"),
             jsonl.text(rec.get("drop_reason"), line, "drop_reason"),
             jsonl.text(rec.get("evidence"), line, "evidence"),
             jsonl.text(rec.get("note"), line, "note"), utc_now()),
        )
    except sqlite3.IntegrityError as err:
        raise FormatError(f"line {line}: {err} (event {rec['event']!r})") from err
