"""Ingest fetch-script output into the listings store.

Fetch scripts write JSONL, format 1. **Line 1 is the run header:**

    {"type": "run", "format": 1, "run_uuid": str, "source": str,
     "kind": "live" | "backfill" | "manual", "started_at": timestamp,
     "status": "ok" | "partial" | "failed", "complete": bool,
     "queries": [{"query": str, "params": object | null,
                  "results_reported": int | null, "results_fetched": int | null}],
     optional: "finished_at", "error", "fetcher", "fetcher_version", "preselection"}

``complete`` is true only when the run enumerated its source fully, so that a
listing's absence from the run means something. **Every later line is one hit,
written before the script's own dedup and filtering:**

    {"type": "hit", "query": str, "source_key": str, "passed_filter": bool,
     optional: "source" (overrides the header's, e.g. two alert sites in one run),
     "observed_at" (default: the run's started_at), "artifact_ref", "company",
     "title", "url", "location", "listing_line", "jd_text", "posted_at",
     "posted_at_field", "comp_min", "comp_max", "comp_currency", "comp_period",
     "level", "filter_reason", "raw" (the verbatim source record)}

``source_key`` must be unique within its source and stable across runs (a board
job id, or a normalized URL). Ingest is one transaction per file and idempotent
on ``run_uuid``: re-ingesting a file is a no-op.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decision_model_eval import jsonl
from decision_model_eval.ats import parse_ats
from decision_model_eval.jsonl import FormatError
from decision_model_eval.store import utc_now

HASH_VERSION = 1
# Normalized content that defines a snapshot. listing_line is excluded: it is
# formatted by the fetch script, so a formatting change is not a content change.
HASHED_FIELDS = (
    "title", "company", "location", "comp_min", "comp_max", "comp_currency",
    "comp_period", "posted_at", "level", "jd_text",
)

_HEADER_REQUIRED = {
    "type", "format", "run_uuid", "source", "kind", "started_at", "status", "complete",
    "queries",
}
_HEADER_OPTIONAL = {"finished_at", "error", "fetcher", "fetcher_version", "preselection"}
_QUERY_KEYS = {"query", "params", "results_reported", "results_fetched"}
_HIT_REQUIRED = {"type", "query", "source_key", "passed_filter"}
_HIT_OPTIONAL = {
    "source", "observed_at", "artifact_ref", "company", "title", "url", "location",
    "listing_line", "jd_text", "posted_at", "posted_at_field", "comp_min", "comp_max",
    "comp_currency", "comp_period", "level", "filter_reason", "raw",
}
_TEXT_FIELDS = (
    "source", "artifact_ref", "company", "title", "url", "location", "listing_line",
    "jd_text", "posted_at_field", "comp_currency", "comp_period", "level", "filter_reason",
)


@dataclass(frozen=True)
class IngestResult:
    run_uuid: str
    already_ingested: bool
    hits: int = 0
    new_listings: int = 0
    new_snapshots: int = 0
    new_sightings: int = 0


def content_hash(hit: dict[str, Any]) -> str:
    """Hash of a hit's normalized content (whitespace collapsed, numbers as floats)."""
    normalized: dict[str, Any] = {}
    for key in HASHED_FIELDS:
        value = hit.get(key)
        if isinstance(value, str):
            value = " ".join(value.split()) or None
        elif isinstance(value, int | float) and not isinstance(value, bool):
            value = float(value)
        normalized[key] = value
    return hashlib.sha256(jsonl.canonical_json(normalized).encode()).hexdigest()


def company_key(name: str) -> str:
    return " ".join(name.split()).casefold()


def _header(record: dict[str, Any], line: int) -> dict[str, Any]:
    if record.get("type") != "run":
        raise FormatError(f"line {line}: the first record must be the run header")
    jsonl.check_keys(record, line, _HEADER_REQUIRED, _HEADER_OPTIONAL)
    if record["format"] != jsonl.FORMAT_VERSION:
        raise FormatError(f"line {line}: unsupported format {record['format']!r}")
    if record["kind"] not in ("live", "backfill", "manual"):
        raise FormatError(f"line {line}: kind must be live, backfill or manual")
    if record["status"] not in ("ok", "partial", "failed"):
        raise FormatError(f"line {line}: status must be ok, partial or failed")
    queries = record["queries"]
    if not isinstance(queries, list) or not queries:
        raise FormatError(f"line {line}: queries must be a non-empty list")
    for query in queries:
        if not isinstance(query, dict):
            raise FormatError(f"line {line}: each query must be an object")
        jsonl.check_keys(query, line, {"query"}, _QUERY_KEYS - {"query"})
        jsonl.required_text(query["query"], line, "queries.query")
        if query.get("params") is not None and not isinstance(query["params"], dict):
            raise FormatError(f"line {line}: queries.params must be an object")
        for key in ("results_reported", "results_fetched"):
            jsonl.integer(query.get(key), line, f"queries.{key}")
    names = [q["query"] for q in queries]
    if len(names) != len(set(names)):
        raise FormatError(f"line {line}: duplicate query in header")
    return {
        "run_uuid": jsonl.required_text(record["run_uuid"], line, "run_uuid"),
        "source": jsonl.required_text(record["source"], line, "source"),
        "kind": record["kind"],
        "started_at": jsonl.timestamp(record["started_at"], line, "started_at"),
        "finished_at": (
            jsonl.timestamp(record["finished_at"], line, "finished_at")
            if record.get("finished_at") is not None else None
        ),
        "status": record["status"],
        "error": jsonl.text(record.get("error"), line, "error"),
        "complete": jsonl.boolean(record["complete"], line, "complete"),
        "fetcher": jsonl.text(record.get("fetcher"), line, "fetcher"),
        "fetcher_version": jsonl.text(record.get("fetcher_version"), line, "fetcher_version"),
        "preselection": jsonl.text(record.get("preselection"), line, "preselection"),
        "queries": queries,
    }


def _hit(record: dict[str, Any], line: int, header: dict[str, Any]) -> dict[str, Any]:
    jsonl.check_keys(record, line, _HIT_REQUIRED, _HIT_OPTIONAL)
    if record["type"] != "hit":
        raise FormatError(f"line {line}: expected a hit record, got {record['type']!r}")
    hit: dict[str, Any] = {key: jsonl.text(record.get(key), line, key) for key in _TEXT_FIELDS}
    hit["query"] = jsonl.required_text(record["query"], line, "query")
    if hit["query"] not in {q["query"] for q in header["queries"]}:
        raise FormatError(f"line {line}: query {hit['query']!r} is not in the run header")
    hit["source_key"] = jsonl.required_text(record["source_key"], line, "source_key")
    hit["source"] = hit["source"] or header["source"]
    hit["passed_filter"] = jsonl.boolean(record["passed_filter"], line, "passed_filter")
    hit["observed_at"] = (
        jsonl.timestamp(record["observed_at"], line, "observed_at")
        if record.get("observed_at") is not None else header["started_at"]
    )
    hit["posted_at"] = (
        jsonl.date_or_timestamp(record["posted_at"], line, "posted_at")
        if record.get("posted_at") is not None else None
    )
    hit["comp_min"] = jsonl.number(record.get("comp_min"), line, "comp_min")
    hit["comp_max"] = jsonl.number(record.get("comp_max"), line, "comp_max")
    raw = record.get("raw")
    if raw is not None and not isinstance(raw, dict):
        raise FormatError(f"line {line}: raw must be an object")
    hit["raw"] = raw
    return hit


def parse_file(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate a whole file before anything is written."""
    records = jsonl.read_records(path)
    header_line, header_record = records[0]
    header = _header(header_record, header_line)
    return header, [_hit(record, line, header) for line, record in records[1:]]


def ingest_file(conn: sqlite3.Connection, path: str | Path) -> IngestResult:
    """Validate and load one fetch-script JSONL file in a single transaction."""
    header, hits = parse_file(path)
    run_uuid = header["run_uuid"]
    if conn.execute("SELECT 1 FROM runs WHERE run_uuid = ?", (run_uuid,)).fetchone():
        return IngestResult(run_uuid=run_uuid, already_ingested=True)

    conn.execute("BEGIN")
    try:
        result = _write(conn, header, hits)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return result


def _write(
    conn: sqlite3.Connection, header: dict[str, Any], hits: list[dict[str, Any]]
) -> IngestResult:
    preselection_id = None
    if header["preselection"] is not None:
        row = conn.execute(
            "SELECT id FROM preselection_versions WHERE name = ?", (header["preselection"],)
        ).fetchone()
        if row is None:
            raise FormatError(
                f"unknown preselection version {header['preselection']!r}; "
                "record it first with `dme preselection add`"
            )
        preselection_id = row["id"]

    run_id = conn.execute(
        "INSERT INTO runs (run_uuid, source, kind, fetcher, fetcher_version, "
        "preselection_version_id, started_at, finished_at, status, error, complete, "
        "ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (header["run_uuid"], header["source"], header["kind"], header["fetcher"],
         header["fetcher_version"], preselection_id, header["started_at"],
         header["finished_at"], header["status"], header["error"], header["complete"],
         utc_now()),
    ).lastrowid

    query_ids: dict[str, int] = {}
    for query in header["queries"]:
        params = query.get("params")
        query_ids[query["query"]] = conn.execute(
            "INSERT INTO run_queries (run_id, query, params, results_reported, "
            "results_fetched) VALUES (?, ?, ?, ?, ?)",
            (run_id, query["query"], jsonl.canonical_json(params) if params else None,
             query.get("results_reported"), query.get("results_fetched")),
        ).lastrowid or 0

    new_listings = new_snapshots = new_sightings = 0
    for hit in hits:
        listing_id, created = _listing(conn, hit)
        new_listings += created
        snapshot_id, created = _snapshot(conn, listing_id, hit)
        new_snapshots += created
        payload_hash = _payload(conn, hit["raw"])
        cursor = conn.execute(
            "INSERT OR IGNORE INTO sightings (run_query_id, listing_id, snapshot_id, "
            "payload_hash, observed_at, artifact_ref, passed_filter, filter_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (query_ids[hit["query"]], listing_id, snapshot_id, payload_hash,
             hit["observed_at"], hit["artifact_ref"], hit["passed_filter"],
             hit["filter_reason"]),
        )
        new_sightings += cursor.rowcount

    return IngestResult(
        run_uuid=header["run_uuid"], already_ingested=False, hits=len(hits),
        new_listings=new_listings, new_snapshots=new_snapshots, new_sightings=new_sightings,
    )


def _listing(conn: sqlite3.Connection, hit: dict[str, Any]) -> tuple[int, int]:
    """Listing id for the hit, creating it if new. First-seen values are kept."""
    company_id = None
    if hit["company"] and hit["company"].strip():
        name = " ".join(hit["company"].split())
        conn.execute(
            "INSERT INTO companies (name, name_key) VALUES (?, ?) "
            "ON CONFLICT (name_key) DO NOTHING",
            (name, company_key(name)),
        )
        company_id = conn.execute(
            "SELECT id FROM companies WHERE name_key = ?", (company_key(name),)
        ).fetchone()["id"]

    existing = conn.execute(
        "SELECT id FROM listings WHERE source = ? AND source_key = ?",
        (hit["source"], hit["source_key"]),
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE listings SET company_id = COALESCE(company_id, ?), url = COALESCE(url, ?) "
            "WHERE id = ?",
            (company_id, hit["url"], existing["id"]),
        )
        return existing["id"], 0

    ats = parse_ats(hit["url"])
    listing_id = conn.execute(
        "INSERT INTO listings (source, source_key, company_id, url, ats, ats_tenant, "
        "ats_job_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (hit["source"], hit["source_key"], company_id, hit["url"],
         ats.ats if ats else None, ats.tenant if ats else None, ats.job_id if ats else None),
    ).lastrowid
    return listing_id or 0, 1


def _snapshot(conn: sqlite3.Connection, listing_id: int, hit: dict[str, Any]) -> tuple[int, int]:
    """Snapshot id for the hit's content, creating it if this content is new."""
    digest = content_hash(hit)
    existing = conn.execute(
        "SELECT id, first_seen_at FROM snapshots "
        "WHERE listing_id = ? AND hash_version = ? AND content_hash = ?",
        (listing_id, HASH_VERSION, digest),
    ).fetchone()
    if existing:
        if hit["observed_at"] < existing["first_seen_at"]:  # e.g. a later backfill
            conn.execute(
                "UPDATE snapshots SET first_seen_at = ? WHERE id = ?",
                (hit["observed_at"], existing["id"]),
            )
        return existing["id"], 0
    snapshot_id = conn.execute(
        "INSERT INTO snapshots (listing_id, hash_version, content_hash, title, "
        "company_name_raw, listing_line, location, comp_min, comp_max, comp_currency, "
        "comp_period, posted_at, posted_at_field, level, jd_text, first_seen_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (listing_id, HASH_VERSION, digest, hit["title"], hit["company"], hit["listing_line"],
         hit["location"], hit["comp_min"], hit["comp_max"], hit["comp_currency"],
         hit["comp_period"], hit["posted_at"], hit["posted_at_field"], hit["level"],
         hit["jd_text"], hit["observed_at"]),
    ).lastrowid
    return snapshot_id or 0, 1


def _payload(conn: sqlite3.Connection, raw: dict[str, Any] | None) -> str | None:
    if raw is None:
        return None
    text = jsonl.canonical_json(raw)
    digest = hashlib.sha256(text.encode()).hexdigest()
    conn.execute("INSERT OR IGNORE INTO payloads (hash, raw) VALUES (?, ?)", (digest, text))
    return digest
