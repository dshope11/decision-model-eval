"""Tests for decision_model_eval.surfaced."""

from __future__ import annotations

import sqlite3
from typing import Any

from conftest import WriteJsonl, hit, run_header

from decision_model_eval.ingest import ingest_file
from decision_model_eval.surfaced import changed_since_judged, surfaced

LAST_WEEK = "2026-09-26T20:00:00Z"
TONIGHT = "2026-10-03T20:00:00Z"


def ingest_at(conn: sqlite3.Connection, path: object, ingested_at: str) -> None:
    result = ingest_file(conn, path)  # type: ignore[arg-type]
    conn.execute("UPDATE runs SET ingested_at = ? WHERE run_uuid = ?",
                 (ingested_at, result.run_uuid))


def test_new_vs_seen_and_rejects_excluded(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    ingest_at(conn, write_jsonl(run_header(run_uuid="last-week"), hit()), LAST_WEEK)
    ingest_at(conn, write_jsonl(
        run_header(run_uuid="tonight"),
        hit(),                                                   # seen last week
        hit(source_key="2", title="Research Engineer"),          # new tonight
        hit(source_key="3", passed_filter=False, filter_reason="not US or remote"),
    ), TONIGHT)
    rows = surfaced(conn, TONIGHT)
    assert [(r.view.title, r.new) for r in rows] == [
        ("Research Engineer", True),
        ("Software Engineer, Machine Learning Systems", False),
    ]


def test_listing_surfaced_by_two_queries_appears_once(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    header = run_header(queries=[{"query": "ml engineer"}, {"query": "research engineer"}])
    ingest_at(conn, write_jsonl(header, hit(), hit(query="research engineer")), TONIGHT)
    assert len(surfaced(conn, TONIGHT)) == 1


def board_run(source: str, run_uuid: str, *hits: dict[str, Any]) -> list[dict[str, Any]]:
    return [run_header(source=source, run_uuid=run_uuid), *hits]


def test_same_getro_job_on_two_boards_appears_once(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    ingest_at(conn, write_jsonl(*board_run("getro:103", "lux", hit())), TONIGHT)
    ingest_at(conn, write_jsonl(*board_run("getro:222", "gc", hit())), TONIGHT)
    (row,) = surfaced(conn, TONIGHT)
    gc_id = conn.execute("SELECT id FROM listings WHERE source = 'getro:222'").fetchone()[0]
    assert (row.source, row.new, row.also) == ("getro:103", True, ((gc_id, "getro:222"),))


def test_board_twin_sighted_earlier_makes_the_posting_seen(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    ingest_at(conn, write_jsonl(*board_run("getro:103", "lux", hit())), LAST_WEEK)
    ingest_at(conn, write_jsonl(*board_run("getro:222", "gc", hit())), TONIGHT)
    (row,) = surfaced(conn, TONIGHT)
    assert (row.source, row.new, row.also) == ("getro:222", False, ())


def test_shared_ats_job_folds_across_sources(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    url = "https://job-boards.greenhouse.io/acmerobotics/jobs/4012345"
    ingest_at(conn, write_jsonl(*board_run("getro:103", "lux", hit(url=url))), TONIGHT)
    ingest_at(conn, write_jsonl(*board_run("waas", "waas", hit(source_key="77", url=url))),
              TONIGHT)
    (row,) = surfaced(conn, TONIGHT)
    assert [source for _, source in row.also] == ["waas"]


def test_ats_tenant_differing_only_in_case_folds(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    job = "00000000-0000-4000-8000-000000000001"
    ingest_at(conn, write_jsonl(*board_run("getro:103", "lux", hit(
        source_key="1", url=f"https://jobs.ashbyhq.com/acmerobotics/{job}"),
        hit(source_key="2", url=f"https://jobs.ashbyhq.com/AcmeRobotics/{job}"),
    )), TONIGHT)
    (row,) = surfaced(conn, TONIGHT)
    assert len(row.also) == 1


def decide_on_latest(conn: sqlite3.Connection, created_at: str,
                     backend: str = "claude-inline") -> None:
    conn.execute(
        "INSERT INTO decisions (snapshot_id, decision_point, backend, output, created_at) "
        "SELECT id, 'D5', ?, '{}', ? FROM latest_snapshots", (backend, created_at))


def add_event(conn: sqlite3.Connection, event: str, at: str) -> None:
    conn.execute("INSERT INTO pipeline_events (listing_id, event, at, recorded_at) "
                 "SELECT id, ?, ?, ? FROM listings", (event, at, at))


def week(run_uuid: str, at: str, **changes: Any) -> list[dict[str, Any]]:
    return [run_header(run_uuid=run_uuid, started_at=at), hit(**changes)]


def test_retitled_listing_is_flagged_against_its_last_decision(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    ingest_at(conn, write_jsonl(*week("last-week", LAST_WEEK)), LAST_WEEK)
    decide_on_latest(conn, LAST_WEEK)
    ingest_at(conn, write_jsonl(*week("tonight", TONIGHT, title="Research Engineer")),
              TONIGHT)
    (row,) = surfaced(conn, TONIGHT)
    (change,) = row.changed
    assert change.text == ("since D5 decision 2026-09-26: title Software "
                           "Engineer, Machine Learning Systems -> Research Engineer")


def test_unchanged_view_or_a_decision_on_the_new_content_is_not_flagged(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    ingest_at(conn, write_jsonl(*week("last-week", LAST_WEEK)), LAST_WEEK)
    decide_on_latest(conn, LAST_WEEK)
    ingest_at(conn, write_jsonl(*week("tonight", TONIGHT, jd_text="New JD text.")),
              TONIGHT)
    assert conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == 2
    assert surfaced(conn, TONIGHT)[0].changed == ()
    ingest_at(conn, write_jsonl(*week("later", TONIGHT, title="Research Engineer")),
              TONIGHT)
    decide_on_latest(conn, TONIGHT)
    assert surfaced(conn, TONIGHT)[0].changed == ()


def test_shadow_backend_decisions_are_not_an_anchor(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    ingest_at(conn, write_jsonl(*week("last-week", LAST_WEEK)), LAST_WEEK)
    decide_on_latest(conn, LAST_WEEK, backend="jev")
    ingest_at(conn, write_jsonl(*week("tonight", TONIGHT, title="Research Engineer")),
              TONIGHT)
    assert surfaced(conn, TONIGHT)[0].changed == ()


def test_event_anchor_outlasts_a_later_decision_on_the_new_content(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    ingest_at(conn, write_jsonl(*week("last-week", LAST_WEEK)), LAST_WEEK)
    add_event(conn, "submitted", "2026-09-26")
    ingest_at(conn, write_jsonl(*week("tonight", TONIGHT, level="entry_level")), TONIGHT)
    decide_on_latest(conn, TONIGHT)
    (row,) = surfaced(conn, TONIGHT)
    assert [c.text for c in row.changed] == [
        "since submitted 2026-09-26: level ? -> entry-level"]
    assert changed_since_judged(conn, row.listing_id, events_only=True) == row.changed[0]


def test_event_before_the_first_sighting_compares_with_the_first_snapshot(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    ingest_at(conn, write_jsonl(*week("last-week", LAST_WEEK)), LAST_WEEK)
    add_event(conn, "submitted", "2026-07-05")
    ingest_at(conn, write_jsonl(*week("tonight", TONIGHT, location="Remote, US")), TONIGHT)
    (change,) = surfaced(conn, TONIGHT)[0].changed
    assert change.fields == (("location", "Palo Alto, CA", "Remote, US"),)


def test_matching_keys_outside_getro_or_across_tenants_do_not_fold(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl
) -> None:
    ingest_at(conn, write_jsonl(*board_run("waas", "waas", hit())), TONIGHT)
    ingest_at(conn, write_jsonl(*board_run("job-alerts", "alerts", hit())), TONIGHT)
    ingest_at(conn, write_jsonl(*board_run("getro:103", "lux", hit(
        source_key="1", url="https://job-boards.greenhouse.io/acme/jobs/4012345"),
        hit(source_key="2", url="https://job-boards.greenhouse.io/zenith/jobs/4012345"),
    )), TONIGHT)
    rows = surfaced(conn, TONIGHT)
    assert len(rows) == 4 and all(row.also == () for row in rows)
