"""What a set of runs let through: the listings to triage, with their store ids.

Fetch scripts print human-readable lines that carry no store identity. Triage
works from this list instead, so every inline judgment can be recorded against
the listing it was made on (``"listing": {"listing_id": N}`` in a record file).
Each listing appears once however many queries or emails surfaced it, and is
marked new when no earlier run had ever sighted it - the baseline diff, done in
code rather than by eye.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Surfaced:
    listing_id: int
    new: bool
    source: str
    company: str | None
    title: str | None
    location: str | None
    url: str | None


def surfaced(conn: sqlite3.Connection, ingested_since: str) -> list[Surfaced]:
    """Listings that passed the script filters in runs ingested at or after
    ``ingested_since`` (ISO 8601 UTC). ``new`` means never sighted before those runs."""
    rows = conn.execute(
        """
        WITH window_runs AS (SELECT id FROM runs WHERE ingested_at >= :since),
        hits AS (
            SELECT DISTINCT si.listing_id
            FROM sightings si JOIN run_queries q ON q.id = si.run_query_id
            WHERE q.run_id IN (SELECT id FROM window_runs) AND si.passed_filter = 1
        ),
        earlier AS (
            SELECT DISTINCT si.listing_id
            FROM sightings si JOIN run_queries q ON q.id = si.run_query_id
            WHERE q.run_id NOT IN (SELECT id FROM window_runs)
        )
        SELECT l.id, l.source, l.url, sn.company_name_raw, sn.title, sn.location,
               l.id NOT IN (SELECT listing_id FROM earlier) AS is_new
        FROM hits h
        JOIN listings l ON l.id = h.listing_id
        JOIN latest_snapshots sn ON sn.listing_id = l.id
        ORDER BY is_new DESC, sn.company_name_raw COLLATE NOCASE, sn.title COLLATE NOCASE
        """,
        {"since": ingested_since},
    ).fetchall()
    return [
        Surfaced(listing_id=r["id"], new=bool(r["is_new"]), source=r["source"],
                 company=r["company_name_raw"], title=r["title"], location=r["location"],
                 url=r["url"])
        for r in rows
    ]
