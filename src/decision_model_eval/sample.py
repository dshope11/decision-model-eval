"""Blind label samples: a seeded random draw, a sheet to fill in, and its records.

The human labels are the control on every model judgment, so they are drawn from
everything a set of runs fetched - filter rejects included, because a sample of
only what the filters let through cannot measure what they wrongly kept out.

The sheet shows the listing view alone (``view.listing_view``, the same view
``dme surfaced`` gives triage): no tier, no filter verdict, no model output. The
labeler fills in one answer per entry, and ``sheet_records`` turns the filled
sheet into a ``dme record`` annotations file. Its batch UUID is derived from the
sample name, decision point and labeler, so recording the same sheet twice is a
no-op.

Sheet format (plain text, edited in any editor)::

    # sample: blind-2026-W41
    # decision_point: D1
    # rubric_version: d1-v2
    # question: Worth reading the full JD? (y / n)
    # ...
    [1] L1234 S5678
        Acme :: Research Engineer :: San Francisco, CA, US
        work: remote ok | level: 3+ yrs | role: eng: ML
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
from decision_model_eval.view import ListingView, listing_view

# Fixed namespace so a sheet's batch UUID is reproducible across machines.
_NAMESPACE = uuid.UUID("6f1d3c2a-7b4e-4d8a-9c5f-2e1a0b9d8c7e")
_ENTRY = re.compile(r"^\[(\d+)\] L(\d+) S(\d+)$")
_ANSWERS = {"y": True, "yes": True, "n": False, "no": False}


@dataclass(frozen=True)
class SampleItem:
    listing_id: int
    snapshot_id: int
    view: ListingView


@dataclass(frozen=True)
class SheetEntry:
    listing_id: int
    snapshot_id: int
    answer: bool
    note: str | None


def population(conn: sqlite3.Connection, ingested_since: str,
               ingested_before: str | None = None) -> list[tuple[int, int]]:
    """(listing_id, latest snapshot_id) for every listing any run ingested at or
    after ``ingested_since`` (and before ``ingested_before``, when given) sighted,
    whether or not it passed the filters."""
    rows = conn.execute(
        """
        SELECT DISTINCT si.listing_id, sn.id AS snapshot_id
        FROM sightings si
        JOIN run_queries q ON q.id = si.run_query_id
        JOIN runs r ON r.id = q.run_id
        JOIN latest_snapshots sn ON sn.listing_id = si.listing_id
        WHERE r.ingested_at >= :since AND (:before IS NULL OR r.ingested_at < :before)
        ORDER BY si.listing_id
        """,
        {"since": ingested_since, "before": ingested_before},
    ).fetchall()
    return [(int(r["listing_id"]), int(r["snapshot_id"])) for r in rows]


def sample_name_used(conn: sqlite3.Connection, name: str) -> bool:
    """Whether labels already carry this sample name. ``dme label`` keys a sheet's
    batch on its name, so a reused name would make the new sheet a silent no-op."""
    return conn.execute("SELECT 1 FROM labels WHERE sample = ? LIMIT 1",
                        (name,)).fetchone() is not None


def sheet_header(path: Path) -> dict[str, str]:
    """The ``# key: value`` lines at the top of a sheet; empty for any other file."""
    meta: dict[str, str] = {}
    with path.open() as f:
        for raw in f:
            if not raw.startswith("#"):
                break
            key, sep, value = raw[1:].partition(":")
            if sep:
                meta[key.strip()] = value.strip()
    return meta if "sample" in meta else {}


def resolve_rubric(meta: dict[str, str], override: str | None) -> str:
    """The rubric version to record: the sheet header's, which a ``--rubric-version``
    may supply for a header-less sheet but never contradict."""
    header = meta.get("rubric_version")
    if header and override and override != header:
        raise ValueError(f"--rubric-version {override} contradicts the sheet's "
                         f"rubric_version {header}")
    rubric = header or override
    if not rubric:
        raise ValueError("the sheet has no '# rubric_version:' header; pass --rubric-version")
    return rubric


def already_sampled(conn: sqlite3.Connection, decision_point: str) -> set[int]:
    """Listing ids that already carry a label from a sample at ``decision_point``."""
    rows = conn.execute(
        """
        SELECT DISTINCT sn.listing_id
        FROM labels la JOIN snapshots sn ON sn.id = la.snapshot_id
        WHERE la.sample IS NOT NULL AND la.decision_point = ?
        """,
        (decision_point,),
    ).fetchall()
    return {int(r["listing_id"]) for r in rows}


def sheet_listing_ids(path: Path) -> set[int]:
    """Listing ids on a sheet, answered or not - for sheets drawn but not yet recorded."""
    ids = set()
    for line in path.read_text().splitlines():
        match = _ENTRY.match(line.strip())
        if match:
            ids.add(int(match.group(2)))
    return ids


def draw(conn: sqlite3.Connection, ingested_since: str, size: int,
         seed: int, exclude: frozenset[int] = frozenset(),
         ingested_before: str | None = None) -> list[SampleItem]:
    """A seeded simple random sample of ``size`` listings from ``population``,
    less the ``exclude`` listing ids, in drawn order. The same store, window, size,
    seed and exclusions give the same draw. Successive draws from the same closed
    window that each exclude the earlier ones add up to one simple random sample
    without replacement - an open window grows when the next run is ingested."""
    pool = [p for p in population(conn, ingested_since, ingested_before)
            if p[0] not in exclude]
    if size > len(pool):
        raise ValueError(f"sample size {size} exceeds the population of {len(pool)}")
    picked = random.Random(seed).sample(pool, size)
    return [SampleItem(lid, sid, listing_view(conn, sid)) for lid, sid in picked]


def write_sheet(items: list[SampleItem], path: Path, *, name: str, decision_point: str,
                rubric_version: str, question: str, since: str, seed: int,
                population_size: int, excluded: int = 0, until: str | None = None) -> None:
    window = f"since {since}" + (f" and before {until}" if until else "")
    drawn = (f"# drawn: {len(items)} of {population_size} listings sighted by runs "
             f"ingested {window}, seed {seed}")
    if excluded:
        drawn += f", {excluded} already sampled excluded"
    header = [
        f"# sample: {name}",
        f"# decision_point: {decision_point}",
        f"# rubric_version: {rubric_version}",
        f"# question: {question}",
        drawn,
        f"# Answer y or n after each '{decision_point}:'; 'note:' is optional.",
        "# Label from the listing lines alone - no JD, no model output.",
        "",
    ]
    body = []
    for k, item in enumerate(items, start=1):
        body += [f"[{k}] L{item.listing_id} S{item.snapshot_id}", f"    {item.view.line}",
                 f"    {' | '.join(item.view.details)}",
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
