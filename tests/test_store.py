"""Tests for decision_model_eval.store - migrations, constraints, backups."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from decision_model_eval.store import backup, connect, migrations, prune_backups, schema_version


def test_new_store_is_at_latest_schema_version(conn: sqlite3.Connection) -> None:
    assert schema_version(conn) == len(migrations()) >= 1


def test_reconnect_does_not_reapply_migrations(tmp_path: Path) -> None:
    path = tmp_path / "listings.db"
    connect(path).close()
    conn = connect(path)  # would fail on CREATE TABLE if re-applied
    assert schema_version(conn) == len(migrations())


def test_foreign_keys_are_enforced(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        conn.execute(
            "INSERT INTO listings (source, source_key, company_id) VALUES ('x', '1', 999)"
        )


def test_tables_are_strict(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO run_queries (run_id, query, results_reported) "
            "VALUES (1, 'q', 'not a number')"
        )


def test_json_columns_are_checked(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        conn.execute("INSERT INTO payloads (hash, raw) VALUES ('h', '{not json')")


def test_pipeline_event_vocabulary_is_checked(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT INTO listings (source, source_key) VALUES ('x', '1')")
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        conn.execute(
            "INSERT INTO pipeline_events (listing_id, event, at, recorded_at) "
            "VALUES (1, 'hired-on-the-spot', '2026-10-03', '2026-10-03T00:00:00Z')"
        )


def test_backup_is_a_complete_readable_copy(conn: sqlite3.Connection, tmp_path: Path) -> None:
    conn.execute("INSERT INTO listings (source, source_key) VALUES ('x', '1')")
    copy = backup(conn, tmp_path / "backups")
    restored = sqlite3.connect(copy)
    assert restored.execute("SELECT COUNT(*) FROM listings").fetchone()[0] == 1
    assert restored.execute("PRAGMA user_version").fetchone()[0] == len(migrations())


def test_prune_keeps_recent_newest_per_day_for_the_window_and_per_month(
    tmp_path: Path,
) -> None:
    names = [
        "listings-20260930T120000Z.db",                                  # September
        "listings-20261008T090000Z.db", "listings-20261008T180000Z.db",  # before the window
        "listings-20261020T090000Z.db", "listings-20261020T180000Z.db",  # in the window
        "listings-20261022T010000Z.db", "listings-20261022T020000Z.db",
        "listings-20261022T030000Z.db", "listings-20261022T040000Z.db",  # today
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"")
    prune_backups(tmp_path, keep_recent=2, keep_days=14, today=date(2026, 10, 22))
    assert sorted(p.name for p in tmp_path.glob("listings-*.db")) == [
        "listings-20260930T120000Z.db",  # newest of September
        "listings-20261020T180000Z.db",  # newest of 10/20; 10/09-10/22 is the window
        "listings-20261022T030000Z.db", "listings-20261022T040000Z.db",  # 2 newest
    ]


def test_prune_keeps_recent_and_newest_per_month(tmp_path: Path) -> None:
    names = [
        "listings-20260701T000000Z.db", "listings-20260715T000000Z.db",  # July
        "listings-20260801T000000Z.db", "listings-20260820T000000Z.db",  # August
        "listings-20260901T000000Z.db", "listings-20260910T000000Z.db",  # September
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"")
    deleted = prune_backups(tmp_path, keep_recent=2)
    kept = sorted(p.name for p in tmp_path.glob("listings-*.db"))
    # 2 most recent (both September) + newest of July and of August
    assert kept == [
        "listings-20260715T000000Z.db", "listings-20260820T000000Z.db",
        "listings-20260901T000000Z.db", "listings-20260910T000000Z.db",
    ]
    assert sorted(p.name for p in deleted) == [
        "listings-20260701T000000Z.db", "listings-20260801T000000Z.db",
    ]
