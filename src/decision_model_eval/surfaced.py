"""What a set of runs let through: the listings to triage, with their store ids.

Fetch scripts print human-readable lines that carry no store identity. Triage
works from this list instead, so every inline judgment can be recorded against
the listing it was made on (``"listing": {"listing_id": N}`` in a record file).
Each posting appears once however many queries, emails or boards surfaced it, and
is marked new when no earlier run had ever sighted it - the baseline diff, done in
code rather than by eye. Each listing is shown through ``view.listing_view``, the
same view the blind label sheets use, so triage and labels see the same fields.

One posting seen through several sources is several listings in the store, never
merged. Here they are folded into one row, keyed on what names the posting
exactly: the getro job id, which every getro board carrying the job shares, and
the ATS identity parsed from the URL. The row shows the oldest listing and names
the others in ``also``.

A listing already judged can come back with new content under the same id: a
retitled req, a new level or band. ``changed`` lists each listing of the row
whose title, location, level or pay moved since it was last judged, so neither a
seen row's earlier tier nor a pipeline row's record silently carries over.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from decision_model_eval.view import ListingView, listing_view

_EARLIER = """
    WITH window_runs AS (SELECT id FROM runs WHERE ingested_at >= :since),
    earlier AS (
        SELECT DISTINCT si.listing_id
        FROM sightings si JOIN run_queries q ON q.id = si.run_query_id
        WHERE q.run_id NOT IN (SELECT id FROM window_runs)
    )
"""


@dataclass(frozen=True)
class Change:
    """How a listing's latest content differs from what it was last judged on."""
    listing_id: int
    since: str  # the judgment: "dropped 2026-10-07", "D5 decision 2026-10-04"
    fields: tuple[tuple[str, str, str], ...]  # (field, then, now)

    @property
    def text(self) -> str:
        return f"since {self.since}: " + "; ".join(
            f"{name} {then} -> {now}" for name, then, now in self.fields)


@dataclass(frozen=True)
class Surfaced:
    listing_id: int
    new: bool
    source: str
    view: ListingView
    url: str | None
    also: tuple[tuple[int, str], ...] = ()  # (listing id, source) of the same posting
    changed: tuple[Change, ...] = ()  # listings of the posting changed since judged


def _compared(view: ListingView) -> dict[str, str]:
    details = dict(d.split(": ", 1) for d in view.details)
    return {"title": view.title, "location": view.location,
            "level": details.get("level", "?"), "pay": details.get("pay", "?")}


def changed_since_judged(conn: sqlite3.Connection, listing_id: int,
                         events_only: bool = False) -> Change | None:
    """The fields of a listing's latest view that differ from the content it was
    judged on: first the content at its latest pipeline event, a step David took on
    what the pipeline row describes; then, unless ``events_only``, the snapshot of
    its latest ``claude-inline`` decision. Shadow backends' decisions are never an
    anchor, since the session must not see that they ran. An event's content is
    the snapshot last sighted on or before its day, or the first one sighted when
    the event predates the store. Only title, location, level and pay count, so a
    JD-text edit is not a change. None when nothing was judged or nothing moved."""
    latest = conn.execute("SELECT id FROM latest_snapshots WHERE listing_id = ?",
                          (listing_id,)).fetchone()
    if latest is None:
        return None
    anchors: list[tuple[str, int]] = []
    event = conn.execute(
        "SELECT event, at FROM pipeline_events WHERE listing_id = ? "
        "ORDER BY at DESC, id DESC LIMIT 1", (listing_id,)).fetchone()
    if event is not None:
        then = conn.execute(
            "SELECT snapshot_id FROM sightings WHERE listing_id = ? "
            "AND substr(observed_at, 1, 10) <= substr(?, 1, 10) "
            "ORDER BY observed_at DESC, id DESC LIMIT 1",
            (listing_id, event["at"])).fetchone()
        if then is None:
            then = conn.execute(
                "SELECT snapshot_id FROM sightings WHERE listing_id = ? "
                "ORDER BY observed_at, id LIMIT 1", (listing_id,)).fetchone()
        anchors.append((f"{event['event']} {event['at'][:10]}", then["snapshot_id"]))
    if not events_only:
        decision = conn.execute(
            """SELECT d.decision_point, d.snapshot_id, d.created_at
               FROM decisions d JOIN snapshots sn ON sn.id = d.snapshot_id
               WHERE sn.listing_id = ? AND d.backend = 'claude-inline'
               ORDER BY d.created_at DESC, d.id DESC LIMIT 1""",
            (listing_id,)).fetchone()
        if decision is not None:
            anchors.append((f"{decision['decision_point']} decision "
                            f"{decision['created_at'][:10]}", decision["snapshot_id"]))
    now = None
    for since, snapshot_id in anchors:
        if snapshot_id == latest["id"]:
            continue
        now = now or _compared(listing_view(conn, latest["id"]))
        then_fields = _compared(listing_view(conn, snapshot_id))
        moved = tuple((name, then_fields[name], now[name])
                      for name in now if then_fields[name] != now[name])
        if moved:
            return Change(listing_id, since, moved)
    return None


def posting_keys(row: sqlite3.Row) -> set[tuple[str, ...]]:
    """Keys naming the posting a listing row is for. A listing with none is only
    ever itself. The ATS key keeps the tenant, so a job id is never matched across
    tenants: a missed fold costs a duplicate row, a wrong one hides a listing. The
    tenant is casefolded, since ATS hosts treat it so and URLs to one job differ
    in case (``ashbyhq.com/acme`` and ``/Acme``)."""
    keys: set[tuple[str, ...]] = set()
    if row["source"].startswith("getro:"):
        keys.add(("getro", row["source_key"]))
    if row["ats"] and row["ats_job_id"]:
        keys.add(("ats", row["ats"], (row["ats_tenant"] or "").casefold(),
                  row["ats_job_id"]))
    return keys


def surfaced(conn: sqlite3.Connection, ingested_since: str) -> list[Surfaced]:
    """Postings that passed the script filters in runs ingested at or after
    ``ingested_since`` (ISO 8601 UTC). ``new`` means no listing of the posting was
    sighted before those runs."""
    params = {"since": ingested_since}
    rows = conn.execute(
        _EARLIER + """,
        hits AS (
            SELECT DISTINCT si.listing_id
            FROM sightings si JOIN run_queries q ON q.id = si.run_query_id
            WHERE q.run_id IN (SELECT id FROM window_runs) AND si.passed_filter = 1
        )
        SELECT l.id, l.source, l.source_key, l.url, l.ats, l.ats_tenant, l.ats_job_id,
               sn.id AS snapshot_id,
               l.id IN (SELECT listing_id FROM earlier) AS sighted_before
        FROM hits h
        JOIN listings l ON l.id = h.listing_id
        JOIN latest_snapshots sn ON sn.listing_id = l.id
        ORDER BY sn.company_name_raw COLLATE NOCASE, sn.title COLLATE NOCASE, l.id
        """,
        params,
    ).fetchall()
    earlier_keys: set[tuple[str, ...]] = set()
    for row in conn.execute(
        _EARLIER + """
        SELECT l.source, l.source_key, l.ats, l.ats_tenant, l.ats_job_id
        FROM listings l WHERE l.id IN (SELECT listing_id FROM earlier)
        """,
        params,
    ):
        earlier_keys |= posting_keys(row)

    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner: dict[tuple[str, ...], int] = {}
    for i, row in enumerate(rows):
        for key in posting_keys(row):
            if key in owner:
                parent[find(i)] = find(owner[key])
            else:
                owner[key] = i
    groups: dict[int, list[int]] = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)

    shown = []
    for members in groups.values():
        members.sort(key=lambda i: rows[i]["id"])
        keys: set[tuple[str, ...]] = set()
        for i in members:
            keys |= posting_keys(rows[i])
        new =not any(rows[i]["sighted_before"] for i in members) and not keys & earlier_keys
        shown.append((members[0], new, members[1:]))
    shown.sort(key=lambda s: (not s[1], s[0]))  # new first, then company and title
    return [
        Surfaced(listing_id=rows[i]["id"], new=new, source=rows[i]["source"],
                 view=listing_view(conn, rows[i]["snapshot_id"]), url=rows[i]["url"],
                 also=tuple((rows[j]["id"], rows[j]["source"]) for j in others),
                 changed=tuple(change for j in (i, *others)
                               if (change := changed_since_judged(conn, rows[j]["id"]))))
        for i, new, others in shown
    ]
