"""``dme`` - command-line interface to the listings store.

    dme init                          create or migrate the store
    dme ingest FILE [FILE ...]        load fetch-script JSONL, then back up
    dme surfaced --since TIMESTAMP    listings those runs let through, with store ids
    dme record FILE                   load decisions / labels / events JSONL
    dme preselection add NAME --config FILE --effective-from DATE [--note TEXT]
    dme preselection list
    dme backup                        write a consistent copy to backup_dir
    dme stats                         row counts per table

Every command reads ``config.yaml`` (``--config`` to override).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from decision_model_eval.config import REPO_ROOT, EvalConfig, load_config
from decision_model_eval.ingest import ingest_file
from decision_model_eval.jsonl import FormatError, timestamp
from decision_model_eval.record import record_file
from decision_model_eval.store import backup, connect, schema_version
from decision_model_eval.surfaced import surfaced

TABLES = (
    "preselection_versions", "runs", "run_queries", "companies", "listings", "snapshots",
    "sightings", "payloads", "record_batches", "decisions", "labels", "pipeline_events",
)


def _open(config: EvalConfig) -> sqlite3.Connection:
    return connect(config.resolved_db_path())


def _init(config: EvalConfig, _: argparse.Namespace) -> int:
    conn = _open(config)
    print(f"{config.resolved_db_path()}: schema version {schema_version(conn)}")
    return 0


def _ingest(config: EvalConfig, args: argparse.Namespace) -> int:
    conn = _open(config)
    for path in args.files:
        result = ingest_file(conn, path)
        if result.already_ingested:
            print(f"{path}: run {result.run_uuid} already ingested - skipped")
        else:
            print(f"{path}: {result.hits} hits -> {result.new_listings} new listings, "
                  f"{result.new_snapshots} new snapshots, {result.new_sightings} sightings")
    if not args.no_backup:
        print(f"backup: {backup(conn, config.backup_dir, config.keep_backups)}")
    return 0


def _surfaced(config: EvalConfig, args: argparse.Namespace) -> int:
    rows = surfaced(_open(config), timestamp(args.since, 0, "--since"))
    for row in rows:
        fields = [f"L{row.listing_id}", "NEW " if row.new else "seen", row.source,
                  row.company or "?", row.title or "?", row.location or "?", row.url or ""]
        print(" | ".join(fields))
    new = sum(row.new for row in rows)
    print(f"[{len(rows)} listings passed the script filters: {new} new, "
          f"{len(rows) - new} seen in earlier runs]", file=sys.stderr)
    return 0


def _record(config: EvalConfig, args: argparse.Namespace) -> int:
    result = record_file(_open(config), args.file)
    if result.already_recorded:
        print(f"{args.file}: batch {result.batch_uuid} already recorded - skipped")
    else:
        print(f"{args.file}: {result.decisions} decisions, {result.labels} labels, "
              f"{result.events} events")
    return 0


def _preselection_add(config: EvalConfig, args: argparse.Namespace) -> int:
    settings = json.loads(Path(args.config_file).read_text())
    if not isinstance(settings, dict):
        raise FormatError(f"{args.config_file}: preselection config must be a JSON object")
    effective_from = date.fromisoformat(args.effective_from).isoformat()
    conn = _open(config)
    conn.execute(
        "INSERT INTO preselection_versions (name, effective_from, config, note) "
        "VALUES (?, ?, ?, ?)",
        (args.name, effective_from, json.dumps(settings, sort_keys=True), args.note),
    )
    print(f"preselection version {args.name!r} recorded, effective {effective_from}")
    return 0


def _preselection_list(config: EvalConfig, _: argparse.Namespace) -> int:
    rows = _open(config).execute(
        "SELECT name, effective_from, note FROM preselection_versions ORDER BY effective_from"
    ).fetchall()
    for row in rows:
        print(f"{row['effective_from']}  {row['name']}  {row['note'] or ''}".rstrip())
    return 0


def _backup(config: EvalConfig, _: argparse.Namespace) -> int:
    print(backup(_open(config), config.backup_dir, config.keep_backups))
    return 0


def _stats(config: EvalConfig, _: argparse.Namespace) -> int:
    conn = _open(config)
    for table in TABLES:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table:22} {count}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dme", description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init", help="create or migrate the store").set_defaults(run=_init)

    ingest = commands.add_parser("ingest", help="load fetch-script JSONL")
    ingest.add_argument("files", nargs="+")
    ingest.add_argument("--no-backup", action="store_true")
    ingest.set_defaults(run=_ingest)

    shown = commands.add_parser("surfaced", help="listings recent runs let through")
    shown.add_argument("--since", required=True,
                       help="runs ingested at or after this ISO 8601 UTC time")
    shown.set_defaults(run=_surfaced)

    record = commands.add_parser("record", help="load decisions / labels / events JSONL")
    record.add_argument("file")
    record.set_defaults(run=_record)

    preselection = commands.add_parser("preselection", help="search-config versions")
    actions = preselection.add_subparsers(dest="action", required=True)
    add = actions.add_parser("add")
    add.add_argument("name")
    add.add_argument("--config-file", required=True, help="JSON object describing the search")
    add.add_argument("--effective-from", required=True, help="YYYY-MM-DD")
    add.add_argument("--note")
    add.set_defaults(run=_preselection_add)
    actions.add_parser("list").set_defaults(run=_preselection_list)

    commands.add_parser("backup", help="write a consistent copy").set_defaults(run=_backup)
    commands.add_parser("stats", help="row counts per table").set_defaults(run=_stats)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.run(load_config(args.config), args))
    except (FormatError, sqlite3.IntegrityError, ValueError) as err:
        print(f"dme: error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
