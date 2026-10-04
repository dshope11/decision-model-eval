"""Tests for decision_model_eval.ingest."""

from __future__ import annotations

import sqlite3

import pytest
from conftest import WriteJsonl, hit, run_header

from decision_model_eval.ingest import content_hash, ingest_file
from decision_model_eval.jsonl import FormatError


def count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_ingest_writes_run_queries_listing_snapshot_sighting_payload(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    result = ingest_file(conn, write_jsonl(run_header(), hit()))
    assert (result.hits, result.new_listings, result.new_snapshots, result.new_sightings) == (
        1, 1, 1, 1)
    query = conn.execute("SELECT * FROM run_queries").fetchone()
    assert (query["results_reported"], query["results_fetched"]) == (2709, 60)
    assert [count(conn, t) for t in ("runs", "companies", "payloads")] == [1, 1, 1]


def test_reingesting_the_same_run_is_a_no_op(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    path = write_jsonl(run_header(), hit())
    ingest_file(conn, path)
    assert ingest_file(conn, path).already_ingested
    assert count(conn, "sightings") == 1


def test_filter_rejects_are_stored_with_their_reason(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    ingest_file(conn, write_jsonl(
        run_header(), hit(passed_filter=False, filter_reason="non-US location")))
    row = conn.execute("SELECT passed_filter, filter_reason FROM sightings").fetchone()
    assert (row["passed_filter"], row["filter_reason"]) == (0, "non-US location")


def test_unchanged_content_reuses_the_snapshot_across_runs(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    ingest_file(conn, write_jsonl(run_header(), hit()))
    # A week later: same content, a volatile raw field changed
    later = write_jsonl(run_header(run_uuid="run-2", started_at="2026-10-10T20:00:00Z"),
                        hit(raw={"id": 95451597, "weight": 3}))
    result = ingest_file(conn, later)
    assert (result.new_listings, result.new_snapshots, result.new_sightings) == (0, 0, 1)
    assert count(conn, "payloads") == 2  # the raw record still changed, and is kept
    presence = conn.execute("SELECT * FROM listing_presence").fetchone()
    assert (presence["first_seen"], presence["last_seen"], presence["n_sightings"]) == (
        "2026-10-03T20:00:00Z", "2026-10-10T20:00:00Z", 2)


def test_changed_content_creates_a_new_snapshot_and_latest_follows_it(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    ingest_file(conn, write_jsonl(run_header(), hit()))
    ingest_file(conn, write_jsonl(
        run_header(run_uuid="run-2", started_at="2026-10-10T20:00:00Z"),
        hit(location="Remote (US)")))
    assert count(conn, "snapshots") == 2
    latest = conn.execute("SELECT location FROM latest_snapshots").fetchone()
    assert latest["location"] == "Remote (US)"


def test_whitespace_and_number_type_do_not_change_the_content_hash() -> None:
    assert content_hash(hit(title="ML  Engineer ", comp_min=120000)) == content_hash(
        hit(title="ML Engineer", comp_min=120000.0))


def test_one_query_hit_twice_in_a_run_is_one_sighting(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    result = ingest_file(conn, write_jsonl(run_header(), hit(), hit()))
    assert (result.hits, result.new_sightings) == (2, 1)


def test_two_queries_hitting_one_listing_are_two_sightings(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    header = run_header(queries=[{"query": "ml engineer"}, {"query": "research engineer"}])
    ingest_file(conn, write_jsonl(header, hit(), hit(query="research engineer")))
    assert (count(conn, "listings"), count(conn, "sightings")) == (1, 2)


def test_rereading_an_alert_email_in_a_later_run_is_not_a_new_observation(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    alert = hit(source="linkedin-alert", source_key="4012345678",
                artifact_ref="<msg-582@linkedin.com>", observed_at="2026-09-27T14:00:00Z")
    header = run_header(source="job-alerts", queries=[{"query": "JobAlerts"}])
    ingest_file(conn, write_jsonl(header, {**alert, "query": "JobAlerts"}))
    second = run_header(run_uuid="run-2", source="job-alerts", queries=[{"query": "JobAlerts"}])
    result = ingest_file(conn, write_jsonl(second, {**alert, "query": "JobAlerts"}))
    assert result.new_sightings == 0
    assert conn.execute("SELECT source FROM listings").fetchone()["source"] == "linkedin-alert"


def test_failed_run_is_recorded_with_its_error(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    ingest_file(conn, write_jsonl(run_header(status="failed", error="HTTP 406")))
    run = conn.execute("SELECT status, error FROM runs").fetchone()
    assert (run["status"], run["error"]) == ("failed", "HTTP 406")


def test_ats_identity_is_parsed_from_the_url(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    ingest_file(conn, write_jsonl(run_header(), hit(
        url="https://job-boards.greenhouse.io/anthropic/jobs/5198255008")))
    listing = conn.execute("SELECT ats, ats_tenant, ats_job_id FROM listings").fetchone()
    assert tuple(listing) == ("greenhouse", "anthropic", "5198255008")


def test_backfill_observation_moves_first_seen_earlier(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    ingest_file(conn, write_jsonl(run_header(), hit()))
    ingest_file(conn, write_jsonl(
        run_header(run_uuid="backfill-1", kind="backfill"),
        hit(observed_at="2026-07-02T00:00:00Z")))
    first_seen = conn.execute("SELECT first_seen_at FROM snapshots").fetchone()[0]
    assert first_seen == "2026-07-02T00:00:00Z"


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([hit()], "first record must be the run header"),
        ([run_header(), hit(typo_key=1)], "unknown keys"),
        ([run_header(), hit(query="not in header")], "not in the run header"),
        ([run_header(started_at="2026-10-03T20:00:00")], "no UTC offset"),
        ([run_header(), hit(passed_filter="yes")], "true or false"),
        ([run_header(preselection="never-recorded")], "unknown preselection"),
    ],
)
def test_bad_input_fails_loudly_and_writes_nothing(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl, records: list[dict[str, object]],
    message: str
) -> None:
    with pytest.raises(FormatError, match=message):
        ingest_file(conn, write_jsonl(*records))
    assert count(conn, "runs") == 0
