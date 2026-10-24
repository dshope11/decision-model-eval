"""Print the D1 reveal for a recorded blind sample: David's D1 labels against the
pipeline's own D1 call, as one Markdown table of every item either side said y.

Usage: python -I reveal_d1.py SHEET [--time 14m54s] [--out SHEET_STEM.reveal.md]

The pipeline's D1 is "passed the script filter AND listing-line tier Medium or
higher" - the definition the W41 reveal used. A filter reject counts as n. Tiers
come from the claude-inline batches recorded after the sample's window opened and
before its labels were (the run's own triage, one batch per pass); a listing those
batches carried only as jd-verified falls back to its latest earlier listing-line
decision, marked "(earlier)". A JD read recorded after the labels but before the next
run is shown "(post-reveal)". Both-n items are counted, not listed. The last column,
"JD vs your call", reads "not read" where no JD was read and is otherwise left blank
for the session to fill in: confirmed / contradicted / JD didn't settle it.

A stratified sample's labels carry ``<sample>/<stratum>``; the headline then adds
the agreement within each stratum. No combined weighted estimate is made here.
"""
import argparse
import json
import re
import sqlite3
from pathlib import Path

from decision_model_eval.config import load_config
from decision_model_eval.sample import sheet_header
from decision_model_eval.view import listing_view

READ_TIERS = {"Very High", "High", "Medium"}
_DRAWN = re.compile(r"since (\S+?)(?: and before (\S+?))?,")
_ENTRY = re.compile(r"^\[(\d+)\] L(\d+) S(\d+)$")
# Disagreements first, then agreements on y.
_GROUPS = [(True, False), (False, True), (True, True)]


def cell(text: str | None) -> str:
    return (text or "").replace("|", "/").replace("\n", " ").strip()


def yn(value: bool) -> str:
    return "y" if value else "n"


def sheet_order(sheet: Path) -> dict[int, int]:
    """listing_id -> the [k] index it carries on the sheet."""
    order = {}
    for line in sheet.read_text().splitlines():
        match = _ENTRY.match(line.strip())
        if match:
            order[int(match.group(2))] = int(match.group(1))
    return order


def run_batches(conn: sqlite3.Connection, since: str, labels_at: str) -> list[sqlite3.Row]:
    """The run's own triage: every claude-inline batch recorded after the window
    opened and before the sample's labels were - one per pass when a run has two."""
    return conn.execute(
        """
        SELECT b.id, b.batch_uuid, b.recorded_at FROM record_batches b
        WHERE b.recorded_at >= ? AND b.recorded_at < ? AND EXISTS (
            SELECT 1 FROM decisions d WHERE d.batch_id = b.id AND d.backend = 'claude-inline')
        ORDER BY b.recorded_at
        """,
        (since, labels_at),
    ).fetchall()


def decision(conn: sqlite3.Connection, listing_id: int, basis: str,
             batch_clause: str, *args) -> dict | None:
    row = conn.execute(
        f"""
        SELECT d.output FROM decisions d JOIN snapshots sn ON sn.id = d.snapshot_id
        WHERE sn.listing_id = ? AND d.backend = 'claude-inline'
          AND json_extract(d.output, '$.basis') = ? AND {batch_clause}
        ORDER BY d.id DESC LIMIT 1
        """,
        (listing_id, basis, *args),
    ).fetchone()
    return json.loads(row["output"]) if row else None


def next_run_at(conn: sqlite3.Connection, before: str | None) -> str:
    """Ingest time of the first run after the sample's window - later decisions belong to it."""
    row = conn.execute("SELECT MIN(ingested_at) AS at FROM runs WHERE ingested_at >= ?",
                       (before or "9999",)).fetchone()
    return row["at"] or "9999"


def filter_verdict(conn: sqlite3.Connection, listing_id: int, since: str,
                   before: str | None) -> tuple[bool, str]:
    rows = conn.execute(
        """
        SELECT si.passed_filter, si.filter_reason
        FROM sightings si JOIN run_queries q ON q.id = si.run_query_id
        JOIN runs r ON r.id = q.run_id
        WHERE si.listing_id = :lid AND r.ingested_at >= :since
          AND (:before IS NULL OR r.ingested_at < :before)
        """,
        {"lid": listing_id, "since": since, "before": before},
    ).fetchall()
    if any(r["passed_filter"] for r in rows):
        return True, ""
    reasons = sorted({r["filter_reason"] for r in rows if r["filter_reason"]})
    return False, "; ".join(reasons) or "no reason recorded"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sheet", type=Path)
    parser.add_argument("--time", help="labeling time, for the headline")
    parser.add_argument("--out", type=Path, help="also write the reveal here "
                        "(by convention <sheet stem>.reveal.md beside the sheet)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing --out")
    args = parser.parse_args()

    meta = sheet_header(args.sheet)
    sample = meta["sample"]
    since, before = _DRAWN.search(meta["drawn"]).groups()
    order = sheet_order(args.sheet)

    conn = sqlite3.connect(Path(load_config().db_path).expanduser())
    conn.row_factory = sqlite3.Row
    next_run = next_run_at(conn, before)
    labels = conn.execute(
        """
        SELECT sn.listing_id, la.snapshot_id, la.value, la.note, la.sample,
               b.recorded_at
        FROM labels la JOIN snapshots sn ON sn.id = la.snapshot_id
        JOIN record_batches b ON b.id = la.batch_id
        WHERE (la.sample = ? OR substr(la.sample, 1, ?) = ?)
          AND la.decision_point = 'D1' AND la.labeler = 'david'
        """,
        (sample, len(sample) + 1, f"{sample}/"),
    ).fetchall()
    if not labels:
        raise SystemExit(f"no recorded D1 labels for {sample}")
    labels_at = min(la["recorded_at"] for la in labels)
    batches = run_batches(conn, since, labels_at)
    in_run = (f"d.batch_id IN ({', '.join(str(b['id']) for b in batches)})"
              if batches else "0")

    items = []
    for la in labels:
        lid = la["listing_id"]
        passed, reason = filter_verdict(conn, lid, since, before)
        line = jd = None
        earlier = post_reveal = False
        if passed:
            line = decision(conn, lid, "listing-line", in_run)
            jd = decision(conn, lid, "jd-verified", in_run)
            if line is None:
                line = decision(conn, lid, "listing-line", "d.batch_id IN (SELECT id FROM "
                                "record_batches WHERE recorded_at < ?)", since)
                earlier = line is not None
        if jd is None:
            # A JD read after the reveal (Phase 3 step 0), in a later batch but before
            # the next run - the pipeline would not have read it on its own.
            jd = decision(conn, lid, "jd-verified", "d.batch_id IN (SELECT id FROM "
                          "record_batches WHERE recorded_at > ? AND recorded_at < ?)",
                          labels_at, next_run)
            post_reveal = jd is not None
        if not passed:
            at_line = f"filter reject: {reason}"
        elif line is None:
            at_line = "passed, untiered"
        else:
            at_line = f"{line['tier']}{' (earlier)' if earlier else ''}: {line['reason']}"
        view = listing_view(conn, la["snapshot_id"])
        location = view.location if len(view.location) <= 40 else view.location[:37] + "..."
        events = conn.execute(
            "SELECT event, note, drop_reason FROM pipeline_events WHERE listing_id = ? AND at >= ?",
            (lid, since[:10]),
        ).fetchall()
        items.append({
            "k": order.get(lid, 0),
            "listing": f"{view.company} :: {view.title} ({location})",
            "you": json.loads(la["value"]) is True,
            "pipe": passed and line is not None and line["tier"] in READ_TIERS,
            "passed": passed,
            "stratum": la["sample"].partition("/")[2],
            "note": la["note"],
            "at_line": at_line,
            "jd": (f"{jd['tier']}{' (post-reveal)' if post_reveal else ''}: {jd['reason']}"
                   if jd else "not read"),
            "events": events,
        })
    items.sort(key=lambda i: i["k"])

    def agreement(group: list[dict]) -> str:
        count = {g: sum(1 for i in group if (i["you"], i["pipe"]) == g)
                 for g in _GROUPS + [(False, False)]}
        return (f"Agree {count[(True, True)] + count[(False, False)]}/{len(group)} "
                f"(both y {count[(True, True)]}, both n {count[(False, False)]}) | "
                f"you y / pipeline n: {count[(True, False)]} | "
                f"you n / pipeline y: {count[(False, True)]}")

    n = len(items)
    you_y = sum(i["you"] for i in items)
    rejected = sum(not i["passed"] for i in items)
    took = f", {args.time}" if args.time else ""
    ids = ", ".join(b["batch_uuid"][:8] for b in batches) or "none"
    out = [
        f"# D1 reveal - {sample}",
        "",
        f"- {n} items{took}, you {you_y} y / {n - you_y} n",
        "- Pipeline's D1 = passed the filter AND listing-line tier Medium or higher "
        f"(decisions: batch{'es' if len(batches) > 1 else ''} {ids})",
        f"- {agreement(items)}",
    ]
    for stratum in sorted({i["stratum"] for i in items if i["stratum"]}):
        group = [i for i in items if i["stratum"] == stratum]
        y = sum(i["you"] for i in group)
        out.append(f"  - {stratum} stratum ({len(group)} items, you {y} y): "
                   f"{agreement(group)}")
    out += [
        f"- Filter: {rejected} rejected, {n - rejected} tiered",
        "",
        "| # | Listing | You / Pipeline | Your note | Pipeline at listing line "
        "| JD read | JD vs your call |",
        "|---|---|---|---|---|---|---|",
    ]
    for group in _GROUPS:
        for i in (i for i in items if (i["you"], i["pipe"]) == group):
            verdict = "not read" if i["jd"] == "not read" else ""
            out.append(f"| {i['k']} | {cell(i['listing'])} | {yn(i['you'])} / {yn(i['pipe'])} "
                       f"| {cell(i['note'])} | {cell(i['at_line'])} | {cell(i['jd'])} "
                       f"| {verdict} |")

    acted = [i for i in items if i["events"]]
    if acted:
        parts = []
        for i in acted:
            for e in i["events"]:
                extra = e["note"] or e["drop_reason"]
                parts.append(f"#{i['k']} {e['event']}{f' ({extra})' if extra else ''}")
        out += ["", "Pipeline events on sample items since the run: " + "; ".join(parts)]

    text = "\n".join(out) + "\n"
    print(text, end="")
    if args.out:
        # The last column is filled by hand afterwards, so never overwrite silently.
        if args.out.exists() and not args.force:
            raise SystemExit(f"{args.out} exists (its last column may be filled in); "
                             "pass --force to overwrite")
        args.out.write_text(text)


if __name__ == "__main__":
    main()
