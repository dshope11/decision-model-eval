"""Tests for decision_model_eval.surfaced."""

from __future__ import annotations

import sqlite3

from conftest import WriteJsonl, hit, run_header

from decision_model_eval.ingest import ingest_file
from decision_model_eval.surfaced import surfaced

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
