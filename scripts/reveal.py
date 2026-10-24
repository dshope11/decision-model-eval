"""Print the reveal for a recorded blind sample: David's D1 labels against the
pipeline's own D1 call, as one Markdown table of every item either side said y,
and - when the sample has a recorded JD sheet - his D2 + D5 labels against the
pipeline's JD reads.

Usage: python -I reveal.py SHEET [--time 14m54s] [--out SHEET_STEM.reveal.md]

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

When the sample also has JD-sheet labels, a second section, "D2 + D5 on the JD",
sets David's D5, gates and gaps against the pipeline's JD read of the same listing
(the run's own, or a post-reveal one). The pipeline records a tier and its gaps,
not per-gate verdicts, so D2 is compared only as "a gate killed it" (any gate fail
of his vs the pipeline's Drop). His D5 scale is very-high / high / medium / drop,
and the pipeline's Low and Drop both count as drop. Agreement counts only rows read
under one rubric version and labels that saw no model output ([rev]).
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
# The pipeline's tiers on David's JD-sheet D5 scale; Low and Drop are both "drop".
PIPELINE_D5 = {"Very High": "very-high", "High": "high", "Medium": "medium",
               "Low": "drop", "Drop": "drop"}
HIGH_PLUS = {"very-high", "high"}
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
    """The latest matching decision's output, with its rubric version as ``_rubric``."""
    row = conn.execute(
        f"""
        SELECT d.output, d.input_payload FROM decisions d
        JOIN snapshots sn ON sn.id = d.snapshot_id
        WHERE sn.listing_id = ? AND d.backend = 'claude-inline'
          AND json_extract(d.output, '$.basis') = ? AND {batch_clause}
        ORDER BY d.id DESC LIMIT 1
        """,
        (listing_id, basis, *args),
    ).fetchone()
    if row is None:
        return None
    output = json.loads(row["output"])
    output["_rubric"] = json.loads(row["input_payload"] or "{}").get("rubric")
    return output


def jd_labels(conn: sqlite3.Connection, sample: str) -> dict[int, dict]:
    """David's JD-sheet labels per listing: D5, D5.gaps and the D2.G<n> gates, the
    rubric they were read under, and which of them saw model output ([rev])."""
    rows = conn.execute(
        """
        SELECT sn.listing_id, la.decision_point, la.value, la.saw_model_output,
               la.rubric_version
        FROM labels la JOIN snapshots sn ON sn.id = la.snapshot_id
        WHERE (la.sample = ? OR substr(la.sample, 1, ?) = ?) AND la.labeler = 'david'
          AND (la.decision_point IN ('D5', 'D5.gaps') OR la.decision_point LIKE 'D2.G%')
        """,
        (sample, len(sample) + 1, f"{sample}/"),
    ).fetchall()
    out: dict[int, dict] = {}
    for r in rows:
        entry = out.setdefault(r["listing_id"], {"gates": {}, "rev": set()})
        point, value = r["decision_point"], json.loads(r["value"])
        entry["rubric"] = r["rubric_version"]
        if r["saw_model_output"]:
            entry["rev"].add(point)
        if point == "D5":
            entry["d5"] = value
        elif point == "D5.gaps":
            entry["gaps"] = value
        else:
            entry["gates"][point.removeprefix("D2.")] = value
    return {lid: e for lid, e in out.items() if "d5" in e}


def gate_summary(gates: dict[str, dict]) -> tuple[str, bool]:
    """(cell text, killed): every gate that did not plainly pass, and whether any
    gate failed outright - a fail verdict, or a publication bar he doesn't clear."""
    marks, killed = [], False
    for name in sorted(gates, key=lambda g: int(g[1:])):
        g = gates[name]
        verdict = g.get("verdict", "pass")
        if g.get("clears") == "n":
            verdict = "fail"
        killed = killed or verdict == "fail"
        mark = f"{name} {verdict}" if verdict != "pass" else ""
        if g.get("ext_fail"):
            mark = f"{name} ext-fail"
        if mark and g.get("unsure"):
            mark += "?"
        if mark:
            marks.append(mark)
    return ("; ".join(marks) or "all pass"), killed


def gaps_cell(gaps: dict | None) -> str:
    if not gaps:
        return "-"
    core, context = gaps.get("core", []), gaps.get("context", [])
    return f"{len(core)} core ({'; '.join(core) or 'none'}) / {len(context)} context"


def jd_section(items: list[dict], mine: dict[int, dict]) -> list[str]:
    """The D2 + D5 comparison: David's JD-sheet labels against the pipeline's JD read
    on the same listing. Agreement counts only rows read under one rubric version
    and labels that saw no model output."""
    rows = []
    for i in sorted((i for i in items if i["lid"] in mine), key=lambda i: i["k"]):
        me, jd = mine[i["lid"]], i["jd_dec"]
        gates, killed = gate_summary(me["gates"])
        pipe = PIPELINE_D5.get(jd["tier"]) if jd else None
        same = jd is not None and jd.get("_rubric") == me["rubric"]
        rows.append({**i, "me": me, "gates": gates, "killed": killed, "pipe_d5": pipe,
                     "same": same})

    def counts(group: list[dict]) -> list[str]:
        d5 = [r for r in group if r["same"] and "D5" not in r["me"]["rev"]]
        exact = sum(r["me"]["d5"] == r["pipe_d5"] for r in d5)
        hi = [(r["me"]["d5"] in HIGH_PLUS, r["pipe_d5"] in HIGH_PLUS) for r in d5]
        hi_agree = sum(a == b for a, b in hi)
        you_only = sum(a and not b for a, b in hi)
        pipe_only = sum(b and not a for a, b in hi)
        kill = [r for r in group if r["same"]
                and not any(p.startswith("D2.") for p in r["me"]["rev"])]
        kill_agree = sum(r["killed"] == (r["jd_dec"]["tier"] == "Drop") for r in kill)
        gapped = [r for r in d5 if r["me"].get("gaps") and r["jd_dec"].get("gaps")]
        gap_agree = sum(len(r["me"]["gaps"].get("core", []))
                        == len(r["jd_dec"]["gaps"].get("core", [])) for r in gapped)
        return [f"D5 exact {exact}/{len(d5)}",
                f"High or above {hi_agree}/{len(d5)} (you only {you_only}, "
                f"pipeline only {pipe_only})",
                f"gate kill {kill_agree}/{len(kill)}",
                f"core-gap count {gap_agree}/{len(gapped)}"]

    skipped = len(rows) - sum(r["same"] for r in rows)
    out = ["", "## D2 + D5 on the JD", "",
           f"- {len(rows)} JDs labeled; compared where the pipeline read the same JD under "
           f"the same rubric ({skipped} not comparable: no pipeline read, or another rubric)",
           f"- {' | '.join(counts(rows))}"]
    for stratum in sorted({r["stratum"] for r in rows if r["stratum"]}):
        group = [r for r in rows if r["stratum"] == stratum]
        out.append(f"  - {stratum} stratum ({len(group)}): {' | '.join(counts(group))}")
    out += ["- Your D5 scale is very-high / high / medium / drop; the pipeline's Low and "
            "Drop both count as drop. Gate kill = any gate fail of yours vs the pipeline's "
            "Drop. Labels marked [rev] saw model output and are left out of the counts.",
            "",
            "| # | Listing | Your D5 | Your gates | Your gaps | Pipeline JD read | "
            "Pipeline gaps |",
            "|---|---|---|---|---|---|---|"]
    for r in rows:
        me, jd = r["me"], r["jd_dec"]
        d5 = me["d5"] + (" [rev]" if "D5" in me["rev"] else "")
        gates = r["gates"] + (" [rev]" if any(p.startswith("D2.") for p in me["rev"]) else "")
        if jd is None:
            read = "not read"
        else:
            read = f"{jd['tier']}{' (post-reveal)' if r['post_reveal'] else ''}: {jd['reason']}"
            if not r["same"]:
                read += f" [rubric {jd.get('_rubric')} vs your {me['rubric']}]"
        out.append(f"| {r['k']} | {cell(r['listing'])} | {d5} | {cell(gates)} "
                   f"| {cell(gaps_cell(me.get('gaps')))} | {cell(read)} "
                   f"| {cell(gaps_cell(jd.get('gaps') if jd else None))} |")
    return out


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
            "lid": lid,
            "jd_dec": jd,
            "post_reveal": post_reveal,
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

    mine = jd_labels(conn, sample)
    if mine:
        out += jd_section(items, mine)

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
