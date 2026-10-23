"""SQLite listings store: connection, schema migrations, consistent backups.

Migrations are numbered SQL files in ``migrations/`` (``0001_init.sql``, ...),
applied in order; the database's ``PRAGMA user_version`` records the last one
applied. Adding a column later is a new migration file - cheap. Data that was
never captured cannot be recovered, which is why capture is complete from the
first schema version.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import UTC, date, datetime, timedelta
from importlib import resources
from pathlib import Path

_MIGRATION_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


def utc_now() -> str:
    """Current time as ISO 8601 UTC text, the store's timestamp format."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def migrations() -> list[tuple[int, str]]:
    """``(version, sql)`` for every migration file, in version order."""
    found = []
    for entry in resources.files("decision_model_eval").joinpath("migrations").iterdir():
        match = _MIGRATION_NAME.match(entry.name)
        if match:
            found.append((int(match.group(1)), entry.read_text()))
    found.sort()
    versions = [v for v, _ in found]
    if versions != list(range(1, len(versions) + 1)):
        raise RuntimeError(f"Migration versions must run 1..n without gaps, got {versions}")
    return found


def connect(path: str | Path) -> sqlite3.Connection:
    """Open the store, creating it if needed, and apply any pending migrations.

    Foreign keys are enforced (SQLite leaves them off by default) and the
    write-ahead log is enabled, so readers never block the weekly ingest.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)  # explicit BEGIN/COMMIT only
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    migrate(conn)
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations, each in its own transaction. Returns the new version."""
    current = schema_version(conn)
    for version, sql in migrations():
        if version <= current:
            continue
        conn.execute("BEGIN")
        try:
            for statement in _statements(sql):
                conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {version}")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        current = version
    return current


def _statements(sql: str) -> list[str]:
    """Split a migration file into complete statements."""
    statements, buffer = [], ""
    for line in sql.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            if buffer.strip():
                statements.append(buffer)
            buffer = ""
    if buffer.strip() and not buffer.strip().startswith("--"):
        raise ValueError(f"Incomplete SQL statement at end of migration: {buffer[:80]!r}")
    return statements


def backup(conn: sqlite3.Connection, backup_dir: str | Path, keep_recent: int = 3,
           keep_days: int = 14) -> Path:
    """Write a consistent copy of the store with ``VACUUM INTO`` and prune old copies.

    The copy is a single self-contained file, safe to sync or upload. The live
    database is not: syncing it mid-write, or apart from its WAL files, can
    corrupt it. Retention is ``prune_backups``'s rule.
    """
    backup_dir = Path(backup_dir).expanduser()
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = backup_dir / f"listings-{stamp}.db"
    conn.execute("VACUUM INTO ?", (str(target),))
    prune_backups(backup_dir, keep_recent, keep_days)
    return target


def prune_backups(backup_dir: Path, keep_recent: int, keep_days: int = 14,
                  today: date | None = None) -> list[Path]:
    """Delete backups outside the retention rule and return them. Kept: the
    ``keep_recent`` newest copies, so a bad write can be rolled back the same day;
    the newest copy of each of the last ``keep_days`` days (UTC, today included),
    since a backup is taken before every store write and several can land in one
    day; and the newest copy of every month."""
    copies = sorted(backup_dir.glob("listings-*.db"), reverse=True)  # newest first
    keep = set(copies[:keep_recent])
    first_day = ((today or datetime.now(UTC).date())
                 - timedelta(days=keep_days - 1)).strftime("%Y%m%d")
    newest_per_day: dict[str, Path] = {}
    newest_per_month: dict[str, Path] = {}
    for copy in copies:
        stamp = copy.name[len("listings-"):]
        newest_per_month.setdefault(stamp[:6], copy)  # YYYYMM
        if stamp[:8] >= first_day:
            newest_per_day.setdefault(stamp[:8], copy)  # YYYYMMDD
    keep |= set(newest_per_day.values()) | set(newest_per_month.values())
    deleted = [copy for copy in copies if copy not in keep]
    for copy in deleted:
        copy.unlink()
    return deleted
