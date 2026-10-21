"""``dme`` - command-line interface to the listings store.

    dme init                          create or migrate the store
    dme ingest FILE [FILE ...]        load fetch-script JSONL, then back up
    dme surfaced --since TIMESTAMP    listings those runs let through, with store ids
    dme record FILE                   load decisions / labels / events JSONL, or a
                                      decision log's listing decisions
    dme replay LOG --backend KEY --out NEW_LOG [--repeats N]
                                      ask a decision log's inputs again, against any
                                      backend, into a new log
    dme sample --since TIMESTAMP [--until TIMESTAMP] --name NAME --out SHEET [--size 30]
               [--seed N] [--exclude SHEET ...] [--include-sampled]
                                      seeded blind draw, written as a label sheet;
                                      listings already labeled in a sample, or on a
                                      sheet beside SHEET, are skipped
    dme label SHEET --labeler NAME [--rubric-version V]
                                      record a filled label sheet; the rubric comes
                                      from the sheet header when it has one
    dme preselection add NAME --config FILE --effective-from DATE [--note TEXT]
    dme preselection list
    dme scifact download              fetch the pinned SciFact release into scifact_dir
    dme scifact run --name NAME [--limit N] [--workers 8]
                                      ask every planned SciFact question into a new log
                                      under scifact_dir/runs/NAME, skipping calls answered
    dme scifact report --name NAME    results.json, summary.md and reliability diagrams
                                      of a complete run, into its report/
    dme backup                        write a consistent copy to backup_dir
    dme stats                         row counts per table

Every command reads ``config.yaml`` (``--config`` to override).
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

from decision_model_eval import jsonl, scifact, scifact_report
from decision_model_eval.backends import BACKENDS
from decision_model_eval.config import REPO_ROOT, EvalConfig, load_config
from decision_model_eval.decision_log import DecisionLog
from decision_model_eval.ingest import ingest_file
from decision_model_eval.jsonl import FormatError, timestamp
from decision_model_eval.record import record_file
from decision_model_eval.replay import replay
from decision_model_eval.sample import (
    already_sampled,
    draw,
    parse_sheet,
    population,
    resolve_rubric,
    sample_name_used,
    sheet_header,
    sheet_listing_ids,
    sheet_records,
    write_sheet,
)
from decision_model_eval.store import backup, connect, schema_version
from decision_model_eval.surfaced import surfaced
from decision_model_eval.view import VIEW_VERSION

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
                  row.view.company, row.view.title, row.view.location,
                  "; ".join(row.view.details), row.url or ""]
        print(" | ".join(fields))
    new = sum(row.new for row in rows)
    print(f"[{len(rows)} listings passed the script filters: {new} new, "
          f"{len(rows) - new} seen in earlier runs; listing view {VIEW_VERSION}]",
          file=sys.stderr)
    return 0


def _record(config: EvalConfig, args: argparse.Namespace) -> int:
    result = record_file(_open(config), args.file)
    if result.already_recorded:
        print(f"{args.file}: batch {result.batch_uuid} already recorded - skipped")
    else:
        skipped = f", {result.skipped} failed attempts skipped" if result.skipped else ""
        print(f"{args.file}: {result.decisions} decisions, {result.labels} labels, "
              f"{result.events} events{skipped}")
    return 0


def _replay(_: EvalConfig, args: argparse.Namespace) -> int:
    source_run = jsonl.read_records(args.log)[0][1].get("batch_uuid")
    run = {"replay_of": str(Path(args.log).resolve()), "source_run_uuid": source_run,
           "backend": args.backend, "model": BACKENDS[args.backend].model_label,
           "repeats": args.repeats}
    with DecisionLog(args.out, origin="dme replay", run=run) as log:
        decisions, failed = replay(args.log, backend=args.backend, log=log,
                                   repeats=args.repeats)
    print(f"{args.out}: {len(decisions)} decisions, {failed} failed "
          f"({args.backend}, {BACKENDS[args.backend].model_label})")
    return 0


QUESTIONS = {"D1": "Worth reading the full JD? (y / n)"}
# The rubric a new sheet is labeled under: the question asked on listing view v2.
RUBRICS = {"D1": f"d1-{VIEW_VERSION}"}


def _sample(config: EvalConfig, args: argparse.Namespace) -> int:
    out = Path(args.out)
    if out.exists():
        raise ValueError(f"{out} exists; a drawn sheet is never overwritten")
    if args.decision_point not in QUESTIONS:
        raise ValueError(f"no question defined for decision point {args.decision_point!r}")
    conn = _open(config)
    since = timestamp(args.since, 0, "--since")
    until = timestamp(args.until, 0, "--until") if args.until else None
    seed = config.random_seed if args.seed is None else args.seed
    # Sheets beside the new one count as drawn whether or not they are recorded yet,
    # so drawing the next sheet before recording this one never repeats it.
    siblings = {p: sheet_header(p) for p in out.parent.glob("*.txt")}
    if sample_name_used(conn, args.name) or any(
            h.get("sample") == args.name for h in siblings.values()):
        raise ValueError(f"sample name {args.name!r} is already used; label batches "
                         "are keyed on it, so a reused name would record nothing")
    pool = {lid for lid, _ in population(conn, since, until)}
    exclude: set[int] = set()
    if not args.include_sampled:
        exclude = already_sampled(conn, args.decision_point)
        for path, header in siblings.items():
            if header.get("decision_point") == args.decision_point:
                exclude |= sheet_listing_ids(path)
    for sheet in args.exclude:
        exclude |= sheet_listing_ids(Path(sheet))
    excluded = len(pool & exclude)
    items = draw(conn, since, args.size, seed, frozenset(exclude), until)
    write_sheet(items, out, name=args.name, decision_point=args.decision_point,
                rubric_version=RUBRICS[args.decision_point],
                question=QUESTIONS[args.decision_point], since=since, seed=seed,
                population_size=len(pool) - excluded, excluded=excluded, until=until)
    print(f"{out}: {len(items)} listings drawn (seed {seed}, "
          f"{excluded} already sampled excluded)")
    return 0


def _label(config: EvalConfig, args: argparse.Namespace) -> int:
    sheet = Path(args.sheet)
    meta, entries = parse_sheet(sheet)
    records = sheet_records(meta, entries, labeler=args.labeler,
                            rubric_version=resolve_rubric(meta, args.rubric_version))
    path = sheet.with_suffix(".labels.jsonl")
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    result = record_file(_open(config), path)
    if result.already_recorded:
        print(f"{sheet}: already recorded (batch {result.batch_uuid}) - skipped")
    else:
        yes = sum(e.answer for e in entries)
        print(f"{sheet}: {result.labels} labels recorded ({yes} y, {len(entries) - yes} n)")
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


def _scifact_download(config: EvalConfig, _: argparse.Namespace) -> int:
    fetched = scifact.download(config.scifact_dir)
    pairs = scifact.load_pairs(config.scifact_dir)
    counts = Counter(pair.label for pair in pairs)
    print(f"{Path(config.scifact_dir).expanduser()}: "
          f"{'downloaded' if fetched else 'already present'}; {len(pairs)} "
          f"{scifact.SPLIT} pairs ("
          + ", ".join(f"{counts[o]} {o}" for o in scifact.VERDICT.options) + ")")
    return 0


def _commit() -> str | None:
    """The checkout's commit, with ``-dirty`` when tracked files have changed."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        dirty = git("status", "--porcelain", "--untracked-files=no")
        return git("rev-parse", "HEAD") + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def _scifact_run(config: EvalConfig, args: argparse.Namespace) -> int:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.name):
        raise ValueError(f"run name {args.name!r}: use letters, digits, '.', '_', '-'")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    run_dir = Path(config.scifact_dir).expanduser() / "runs" / args.name
    settings = scifact.run_config(backends=config.backends, seed=config.random_seed,
                                  limit=args.limit)
    done = scifact.answered(run_dir, settings)
    planned = scifact.plan(scifact.load_pairs(config.scifact_dir), backends=config.backends,
                           seed=config.random_seed, limit=args.limit)
    calls = [call for call in planned if call.key not in done]
    if not calls:
        print(f"{run_dir}: all {len(planned)} calls answered")
        return 0
    path = run_dir / f"log-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.jsonl"
    with DecisionLog(path, origin="dme scifact run",
                     run={**settings, "commit": _commit()}) as log:
        ok, failed = scifact.run(calls, log=log, workers=args.workers)
    cost = sum(record.get("cost_usd") or 0 for _, record in jsonl.read_records(path)[1:])
    print(f"{path}: {ok} answered, {failed} failed, {len(done)} answered before; "
          f"${cost:.4f}")
    if failed:
        print(f"run it again to retry the {failed} failed", file=sys.stderr)
    return 1 if failed else 0


def _scifact_report(config: EvalConfig, args: argparse.Namespace) -> int:
    run_dir = Path(config.scifact_dir).expanduser() / "runs" / args.name
    _, text, written = scifact_report.report(run_dir, config.scifact_dir)
    print(text)
    for path in written:
        print(path)
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

    again = commands.add_parser("replay", help="ask a decision log's inputs again")
    again.add_argument("log")
    again.add_argument("--backend", required=True, choices=list(BACKENDS))
    again.add_argument("--out", required=True, help="new decision log; never overwritten")
    again.add_argument("--repeats", type=int, default=1)
    again.set_defaults(run=_replay)

    sample = commands.add_parser("sample", help="seeded blind draw as a label sheet")
    sample.add_argument("--since", required=True,
                        help="draw from runs ingested at or after this ISO 8601 UTC time")
    sample.add_argument("--until",
                        help="and before this one - closes the window, so batches drawn "
                             "after the next run is ingested still share a population")
    sample.add_argument("--name", required=True,
                        help="sample name, e.g. blind-2026-W40; must be new")
    sample.add_argument("--out", required=True, help="sheet to write; never overwritten")
    sample.add_argument("--size", type=int, default=30)
    sample.add_argument("--seed", type=int, help="defaults to random_seed in config.yaml")
    sample.add_argument("--decision-point", default="D1")
    sample.add_argument("--exclude", action="append", default=[], metavar="SHEET",
                        help="also exclude a sheet's listings (drawn, not yet labeled); "
                             "repeatable")
    sample.add_argument("--include-sampled", action="store_true",
                        help="do not exclude listings already labeled in a sample or on "
                             "a sheet beside --out")
    sample.set_defaults(run=_sample)

    label = commands.add_parser("label", help="record a filled label sheet")
    label.add_argument("sheet")
    label.add_argument("--labeler", required=True)
    label.add_argument("--rubric-version",
                       help="for a sheet without a '# rubric_version:' header; an error "
                            "if it contradicts the header")
    label.set_defaults(run=_label)

    preselection = commands.add_parser("preselection", help="search-config versions")
    actions = preselection.add_subparsers(dest="action", required=True)
    add = actions.add_parser("add")
    add.add_argument("name")
    add.add_argument("--config-file", required=True, help="JSON object describing the search")
    add.add_argument("--effective-from", required=True, help="YYYY-MM-DD")
    add.add_argument("--note")
    add.set_defaults(run=_preselection_add)
    actions.add_parser("list").set_defaults(run=_preselection_list)

    control = commands.add_parser("scifact", help="the SciFact control run")
    steps = control.add_subparsers(dest="step", required=True)
    steps.add_parser("download").set_defaults(run=_scifact_download)
    go = steps.add_parser("run", help="ask every planned question; resumes a named run")
    go.add_argument("--name", required=True,
                    help="the run's directory under scifact_dir/runs; reused to resume")
    go.add_argument("--limit", type=int, help="only the first N pairs per question (smoke)")
    go.add_argument("--workers", type=int, default=8)
    go.set_defaults(run=_scifact_run)
    tables = steps.add_parser("report", help="analyze a complete run into its report/")
    tables.add_argument("--name", required=True)
    tables.set_defaults(run=_scifact_report)

    commands.add_parser("backup", help="write a consistent copy").set_defaults(run=_backup)
    commands.add_parser("stats", help="row counts per table").set_defaults(run=_stats)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.run(load_config(args.config), args))
    except (FormatError, sqlite3.IntegrityError, ValueError, FileExistsError,
            RuntimeError) as err:
        print(f"dme: error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
