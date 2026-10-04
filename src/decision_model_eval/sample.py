"""Blind label samples: a seeded random draw, a sheet to fill in, and its records.

The human labels are the control on every model judgment, so they are drawn from
everything a set of runs fetched - filter rejects included, because a sample of
only what the filters let through cannot measure what they wrongly kept out.

The sheet shows the listing line alone: no tier, no filter verdict, no model
output. The labeler fills in one answer per entry, and ``sheet_records`` turns the
filled sheet into a ``dme record`` annotations file. Its batch UUID is derived
from the sample name, decision point and labeler, so recording the same sheet
twice is a no-op.

Sheet format (plain text, edited in any editor)::

    # sample: blind-2026-W40
    # decision_point: D1
    # question: Worth reading the full JD? (y / n)
    # ...
    [1] L1234 S5678
        Acme :: Research Engineer :: San Francisco, CA
        D1:
        note:

The answer goes after ``D1:`` (``y`` or ``n``); ``note:`` is optional.
"""

from __future__ import annotations

import random
import re
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decision_model_eval.jsonl import FormatError

# Fixed namespace so a sheet's batch UUID is reproducible across machines.
_NAMESPACE = uuid.UUID("6f1d3c2a-7b4e-4d8a-9c5f-2e1a0b9d8c7e")
_ENTRY = re.compile(r"^\[(\d+)\] L(\d+) S(\d+)$")
_ANSWERS = {"y": True, "yes": True, "n": False, "no": False}


@dataclass(frozen=True)
class SampleItem:
    listing_id: int
    snapshot_id: int
    line: str


@dataclass(frozen=True)
class SheetEntry:
    listing_id: int
    snapshot_id: int
    answer: bool
    note: str | None


def population(conn: sqlite3.Connection, ingested_since: str) -> list[tuple[int, int]]:
    """(listing_id, latest snapshot_id) for every listing any run ingested at or
    after ``ingested_since`` sighted, whether or not it passed the filters."""
    rows = conn.execute(
        """
        SELECT DISTINCT si.listing_id, sn.id AS snapshot_id
        FROM sightings si
        JOIN run_queries q ON q.id = si.run_query_id
        JOIN runs r ON r.id = q.run_id
        JOIN latest_snapshots sn ON sn.listing_id = si.listing_id
        WHERE r.ingested_at >= ?
        ORDER BY si.listing_id
        """,
        (ingested_since,),
    ).fetchall()
    return [(int(r["listing_id"]), int(r["snapshot_id"])) for r in rows]


def draw(conn: sqlite3.Connection, ingested_since: str, size: int,
         seed: int) -> list[SampleItem]:
    """A seeded simple random sample of ``size`` listings from ``population``,
    in drawn order. The same store, window, size and seed give the same draw."""
    pool = population(conn, ingested_since)
    if size > len(pool):
        raise ValueError(f"sample size {size} exceeds the population of {len(pool)}")
    picked = random.Random(seed).sample(pool, size)
    return [SampleItem(lid, sid, _listing_line(conn, sid)) for lid, sid in picked]


def _listing_line(conn: sqlite3.Connection, snapshot_id: int) -> str:
    row = conn.execute(
        "SELECT company_name_raw, title, location FROM snapshots WHERE id = ?",
        (snapshot_id,),
    ).fetchone()
    parts = [row["company_name_raw"], row["title"], row["location"]]
    return " :: ".join(p if p else "?" for p in parts)


def write_sheet(items: list[SampleItem], path: Path, *, name: str, decision_point: str,
                question: str, since: str, seed: int, population_size: int) -> None:
    header = [
        f"# sample: {name}",
        f"# decision_point: {decision_point}",
        f"# question: {question}",
        f"# drawn: {len(items)} of {population_size} listings sighted by runs "
        f"ingested since {since}, seed {seed}",
        f"# Answer y or n after each '{decision_point}:'; 'note:' is optional.",
        "# Label from the line alone - no JD, no model output.",
        "",
    ]
    body = []
    for k, item in enumerate(items, start=1):
        body += [f"[{k}] L{item.listing_id} S{item.snapshot_id}", f"    {item.line}",
                 f"    {decision_point}:", "    note:", ""]
    path.write_text("\n".join(header + body))


def parse_sheet(path: Path) -> tuple[dict[str, str], list[SheetEntry]]:
    """Header fields and filled entries. Every entry must carry a y/n answer."""
    meta: dict[str, str] = {}
    entries: list[SheetEntry] = []
    current: dict[str, Any] | None = None
    missing: list[str] = []

    def close() -> None:
        if current is None:
            return
        if current.get("answer") is None:
            missing.append(f"[{current['k']}]")
        else:
            entries.append(SheetEntry(current["listing_id"], current["snapshot_id"],
                                      current["answer"], current.get("note")))

    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if line.startswith("#"):
            key, sep, value = line[1:].partition(":")
            if sep and current is None:
                meta[key.strip()] = value.strip()
            continue
        match = _ENTRY.match(line)
        if match:
            close()
            current = {"k": match.group(1), "listing_id": int(match.group(2)),
                       "snapshot_id": int(match.group(3))}
            continue
        if current is None or not line:
            continue
        point = meta.get("decision_point", "")
        if point and line.startswith(f"{point}:"):
            value = line[len(point) + 1:].strip().lower()
            if value:
                if value not in _ANSWERS:
                    raise FormatError(f"{path}:{number}: answer must be y or n, got {value!r}")
                current["answer"] = _ANSWERS[value]
        elif line.startswith("note:"):
            current["note"] = line[len("note:"):].strip() or None
    close()
    for key in ("sample", "decision_point"):
        if key not in meta:
            raise FormatError(f"{path}: header is missing '# {key}:'")
    if missing:
        raise FormatError(f"{path}: no answer for {', '.join(missing)}")
    return meta, entries


def sheet_records(meta: dict[str, str], entries: list[SheetEntry], *, labeler: str,
                  rubric_version: str | None) -> list[dict[str, Any]]:
    """A ``dme record`` annotations file for a filled sheet. The labels record
    ``saw_model_output: false`` - the sheet never shows model output."""
    name, point = meta["sample"], meta["decision_point"]
    batch = uuid.uuid5(_NAMESPACE, f"{name}|{point}|{labeler}")
    records: list[dict[str, Any]] = [
        {"type": "batch", "format": 1, "batch_uuid": str(batch),
         "origin": f"dme label {name}"},
    ]
    for entry in entries:
        record: dict[str, Any] = {
            "type": "label", "listing": {"listing_id": entry.listing_id},
            "snapshot_id": entry.snapshot_id, "decision_point": point,
            "labeler": labeler, "value": entry.answer, "saw_model_output": False,
            "sample": name,
        }
        if rubric_version:
            record["rubric_version"] = rubric_version
        if entry.note:
            record["note"] = entry.note
        records.append(record)
    return records
