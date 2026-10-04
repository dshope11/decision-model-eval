"""Tests for decision_model_eval.record."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest
from conftest import WriteJsonl, hit, run_header

from decision_model_eval.ingest import ingest_file
from decision_model_eval.jsonl import FormatError
from decision_model_eval.record import record_file

LISTING = {"source": "getro:103", "source_key": "95451597"}
BATCH = {"type": "batch", "format": 1, "batch_uuid": "batch-1", "origin": "/discovery test"}


@pytest.fixture
def ingested(conn: sqlite3.Connection, write_jsonl: WriteJsonl) -> sqlite3.Connection:
    ingest_file(conn, write_jsonl(run_header(), hit()))
    return conn


def decision(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "type": "decision", "listing": LISTING, "decision_point": "D1",
        "backend": "claude-inline", "model": "claude-opus-5-5",
        "output": {"read_full_jd": True, "tier": "High", "basis": "listing-line"},
    }
    record.update(overrides)
    return record


def test_records_decision_label_and_event(
    ingested: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    result = record_file(ingested, write_jsonl(
        BATCH,
        decision(),
        {"type": "label", "listing": LISTING, "decision_point": "D1", "labeler": "david",
         "value": True, "sample": "blind-2026-W41", "saw_model_output": False},
        {"type": "event", "listing": LISTING, "event": "promoted", "at": "2026-10-03"},
    ))
    assert (result.decisions, result.labels, result.events) == (1, 1, 1)
    row = ingested.execute("SELECT output, batch_id FROM decisions").fetchone()
    assert json.loads(row["output"])["tier"] == "High"
    assert row["batch_id"] is not None


def test_listing_can_be_referenced_by_store_id(
    ingested: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    listing_id = ingested.execute("SELECT id FROM listings").fetchone()[0]
    record_file(ingested, write_jsonl(BATCH, decision(listing={"listing_id": listing_id})))
    assert ingested.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    with pytest.raises(FormatError, match="no listing listing_id 999"):
        record_file(ingested, write_jsonl(
            {**BATCH, "batch_uuid": "batch-2"}, decision(listing={"listing_id": 999})))


def test_rerecording_a_batch_is_a_no_op(
    ingested: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    path = write_jsonl(BATCH, decision())
    record_file(ingested, path)
    assert record_file(ingested, path).already_recorded
    assert ingested.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1


def test_decision_attaches_to_the_latest_snapshot(
    ingested: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    ingest_file(ingested, write_jsonl(
        run_header(run_uuid="run-2", started_at="2026-10-10T20:00:00Z"),
        hit(location="Remote (US)")))
    record_file(ingested, write_jsonl(BATCH, decision()))
    location = ingested.execute(
        "SELECT sn.location FROM decisions d JOIN snapshots sn ON sn.id = d.snapshot_id"
    ).fetchone()[0]
    assert location == "Remote (US)"


def test_unknown_listing_rolls_back_the_whole_batch(
    ingested: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    missing = decision(listing={"source": "getro:103", "source_key": "nope"})
    with pytest.raises(FormatError, match="ingest its fetch run first"):
        record_file(ingested, write_jsonl(BATCH, decision(), missing))
    assert ingested.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0
    assert ingested.execute("SELECT COUNT(*) FROM record_batches").fetchone()[0] == 0


def test_unknown_event_is_rejected(
    ingested: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    bad = {"type": "event", "listing": LISTING, "event": "ghosted", "at": "2026-10-03"}
    with pytest.raises(FormatError, match="ghosted"):
        record_file(ingested, write_jsonl(BATCH, bad))
