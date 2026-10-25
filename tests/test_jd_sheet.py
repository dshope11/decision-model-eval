"""Tests for decision_model_eval.jd_sheet. The sheet, its gates and its fields are
made up; only the format follows a real JD sheet."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from conftest import WriteJsonl, hit, run_header

from decision_model_eval.cli import main
from decision_model_eval.ingest import ingest_file
from decision_model_eval.jd_sheet import jd_sheet_records, parse_jd_sheet
from decision_model_eval.jsonl import FormatError
from decision_model_eval.sample import sheet_batch, sheet_header

GATES = "\n".join(f"    G{n} gate-{n}: pass" for n in range(1, 8))

SHEET = """\
# sample: blind-2026-W99
# decision_point: D2+D5
# rubric_version: d2-v1
# EXAMPLE (comment lines are never answers)
#   G1 north: fail

[1] L{a} S{sa}   (your D1: y)
    Acme Robotics :: Software Engineer :: Palo Alto, CA
    your D1 note: check the stack
    JD source: store
    ----- JD, verbatim -----
    G1 north: fail
    D5: drop
    note: JD text, not an answer
    ----- end JD -----
    G1 north: pass
    G2 east: fail? - "a quoted requirement"
    G3 south: kind=a venue need=preferred meets= n
    G4 west: pass (strict) | ext-fail: a requirement with no gate [rev]
    G5 up: pass? pref-only, "nice to have"
    G6 down: pass [rev; was: fail?] reread it
    G7 around: held=Python, C++ | not=Rust  tier-drop (1 of 3 not held) [rev]
    D5: medium
    D1-check: partly
    note: the main reason
    off-rubric: small team
    opened-link (pay was listed)

[2] L{b} S{sb}   (your D1: y)
    Zenith Labs :: Research Engineer :: Remote
    ----- JD, verbatim -----
    A short JD.
    ----- end JD -----
{gates}
    D5: high
    D1-check: y
    note:
"""


@pytest.fixture
def store(conn: sqlite3.Connection, write_jsonl: WriteJsonl) -> sqlite3.Connection:
    ingest_file(conn, write_jsonl(run_header(), hit(), hit(source_key="2")))
    conn.commit()
    return conn


def write_sheet(store: sqlite3.Connection, path: Path, sheet: str = SHEET) -> Path:
    (a, sa), (b, sb) = store.execute(
        "SELECT listing_id, id FROM latest_snapshots ORDER BY listing_id").fetchall()
    path.write_text(sheet.format(a=a, sa=sa, b=b, sb=sb, gates=GATES))
    return path


def test_entries_read_into_gate_d5_and_d1_check_labels(
    store: sqlite3.Connection, tmp_path: Path,
) -> None:
    first, second = parse_jd_sheet(write_sheet(store, tmp_path / "s.txt"), "d2-v1")
    opened = "opened-link (pay was listed)"
    assert [(x.decision_point, x.value, x.note, x.saw_model_output) for x in first.labels] == [
        ("D2.G1", {"gate": "north", "verdict": "pass"}, opened, False),
        ("D2.G2", {"gate": "east", "verdict": "fail", "unsure": True},
         f'"a quoted requirement"; {opened}', False),
        ("D2.G3", {"gate": "south", "kind": "a venue", "need": "preferred", "meets": "n"},
         opened, False),
        ("D2.G4", {"gate": "west", "verdict": "pass", "ext_fail": "a requirement with no gate"},
         f"(strict); [rev]; {opened}", True),
        ("D2.G5", {"gate": "up", "verdict": "pass", "unsure": True, "pref_only": True},
         f'pref-only, "nice to have"; {opened}', False),
        ("D2.G6", {"gate": "down", "verdict": "pass"},
         f"reread it; [rev; was: fail?]; {opened}", True),
        ("D2.G7", {"gate": "around", "verdict": "tier-drop", "held": ["Python", "C++"],
                   "not": ["Rust"]}, f"(1 of 3 not held); [rev]; {opened}", True),
        ("D5", "medium", f"the main reason; off-rubric: small team; {opened}", False),
        ("D1-check", "partly", opened, False),
    ]
    assert [(x.decision_point, x.value, x.note) for x in second.labels[-2:]] == [
        ("D5", "high", None), ("D1-check", "y", None)]


def test_list_fields_may_be_empty(store: sqlite3.Connection, tmp_path: Path) -> None:
    sheet = SHEET.replace("{gates}", GATES.replace(
        "G7 gate-7: pass", "G7 gate-7: held= | not=  pass (none listed)"))
    _, second = parse_jd_sheet(write_sheet(store, tmp_path / "s.txt", sheet), "d2-v1")
    assert second.labels[6].value == {"gate": "gate-7", "verdict": "pass",
                                      "held": [], "not": []}


def test_cli_records_once_and_only_rev_lines_saw_model_output(
    store: sqlite3.Connection, tmp_path: Path,
) -> None:
    sheet = write_sheet(store, tmp_path / "blind-2026-W99.d2.txt")
    config = tmp_path / "config.yaml"
    config.write_text(f"db_path: {tmp_path / 'store' / 'listings.db'}\n"
                      f"backup_dir: {tmp_path / 'backups'}\n")
    args = ["--config", str(config), "label", str(sheet), "--labeler", "david"]
    assert main(args) == 0
    assert main(args) == 0  # a no-op the second time
    assert (tmp_path / "blind-2026-W99.d2.labels.jsonl").exists()
    rows = store.execute(
        "SELECT decision_point, value, saw_model_output, sample, rubric_version FROM labels"
    ).fetchall()
    assert len(rows) == 18
    assert {(r["sample"], r["rubric_version"]) for r in rows} == {("blind-2026-W99", "d2-v1")}
    assert sorted(r["decision_point"] for r in rows if r["saw_model_output"]) == [
        "D2.G4", "D2.G6", "D2.G7"]
    assert json.loads(next(r["value"] for r in rows if r["decision_point"] == "D5")) == "medium"


def test_batch_differs_from_the_d1_sheet_of_the_same_sample(
    store: sqlite3.Connection, tmp_path: Path,
) -> None:
    sheet = write_sheet(store, tmp_path / "s.txt")
    meta = sheet_header(sheet)
    records = jd_sheet_records(meta, parse_jd_sheet(sheet, "d2-v1"), labeler="david",
                               rubric_version="d2-v1")
    d1_batch = sheet_batch(meta["sample"], "D1", "david")
    assert records[0]["batch_uuid"] != d1_batch["batch_uuid"]


SHEET_V2 = (
    SHEET.replace("# rubric_version: d2-v1", "# rubric_version: d2-v2")
    .replace("    G6 down: pass [rev; was: fail?] reread it\n",
             "    G6 down: cap-medium - \"10+ years\"\n")
    .replace("    D5: medium\n",
             "    gaps: core= a duty, another (pref) | context= startup years\n    D5: medium\n")
    .replace("    D5: high\n", "    gaps: core= | context=\n    D5: high\n")
    .replace("    ----- end JD -----\n",
             "    ----- end JD -----\n    # G1 north    the criteria, reprinted\n")
)


def test_d2_v2_reads_gaps_and_cap_medium_and_skips_reprinted_criteria(
    store: sqlite3.Connection, tmp_path: Path,
) -> None:
    first, second = parse_jd_sheet(write_sheet(store, tmp_path / "s.txt", SHEET_V2), "d2-v2")
    labels = {label.decision_point: label for label in first.labels}
    assert labels["D2.G6"].value == {"gate": "down", "verdict": "cap-medium"}
    assert labels["D5.gaps"].value == {"core": ["a duty", "another (pref)"],
                                       "context": ["startup years"]}
    assert "reprinted" not in (labels["D5"].note or "")
    assert [label.decision_point for label in second.labels][-3:] == ["D5.gaps", "D5",
                                                                     "D1-check"]
    assert second.labels[-3].value == {"core": [], "context": []}


def test_d2_v3_takes_low_and_reads_dash_or_none_as_an_empty_list(
    store: sqlite3.Connection, tmp_path: Path,
) -> None:
    sheet = (SHEET_V2.replace("# rubric_version: d2-v2", "# rubric_version: d2-v3")
             .replace("gaps: core= | context=", "gaps: core= - | context= none")
             .replace("    D5: high\n", "    D5: low\n"))
    _, second = parse_jd_sheet(write_sheet(store, tmp_path / "s.txt", sheet), "d2-v3")
    labels = {label.decision_point: label for label in second.labels}
    assert labels["D5.gaps"].value == {"core": [], "context": []}
    assert labels["D5"].value == "low"
    with pytest.raises(FormatError):
        parse_jd_sheet(write_sheet(store, tmp_path / "s.txt", sheet.replace(
            "# rubric_version: d2-v3", "# rubric_version: d2-v2")), "d2-v2")


def test_d2_v2_needs_a_gaps_line_and_d2_v1_never_reads_one(
    store: sqlite3.Connection, tmp_path: Path,
) -> None:
    sheet = SHEET_V2.replace("    gaps: core= | context=\n", "")
    with pytest.raises(FormatError, match=r"no answer for \[2\] gaps"):
        parse_jd_sheet(write_sheet(store, tmp_path / "s.txt", sheet), "d2-v2")
    sheet = SHEET_V2.replace("gaps: core= | context=", "gaps: none")
    with pytest.raises(FormatError, match="gaps takes fields"):
        parse_jd_sheet(write_sheet(store, tmp_path / "s.txt", sheet), "d2-v2")
    v1 = SHEET.replace("    D5: high\n", "    gaps: core= x | context=\n    D5: high\n")
    _, second = parse_jd_sheet(write_sheet(store, tmp_path / "s.txt", v1), "d2-v1")
    assert "D5.gaps" not in {label.decision_point for label in second.labels}


def test_unknown_rubric_version_is_refused(store: sqlite3.Connection, tmp_path: Path) -> None:
    with pytest.raises(FormatError, match="no JD-sheet format for rubric_version 'd2-v9'"):
        parse_jd_sheet(write_sheet(store, tmp_path / "s.txt"), "d2-v9")


def test_missing_answers_are_reported_together(
    store: sqlite3.Connection, tmp_path: Path,
) -> None:
    sheet = SHEET.replace("    G3 south: kind=a venue need=preferred meets= n\n", "")
    sheet = sheet.replace("    D5: high\n", "")
    with pytest.raises(FormatError, match=r"no answer for \[1\] G3; \[2\] D5"):
        parse_jd_sheet(write_sheet(store, tmp_path / "s.txt", sheet), "d2-v1")


@pytest.mark.parametrize(("old", "new", "error"), [
    ("G1 north: pass\n", "G1 north: maybe\n", "expected a verdict"),
    ("G1 north: pass\n", "G1 north:\n", "no answer"),
    ("G1 north: pass\n", "G2 north: pass\n", "G2 answered twice"),
    ("G1 north: pass\n", "G8 north: pass\n", "has gates G1-G7"),
    ("D5: medium\n", "D5: great\n", "answer must be one of"),
    ("----- end JD -----\n{gates}", "{gates}", "JD block not closed"),
])
def test_malformed_lines_are_errors(
    store: sqlite3.Connection, tmp_path: Path, old: str, new: str, error: str,
) -> None:
    sheet = SHEET.replace(old, new, 1)
    with pytest.raises(FormatError, match=error):
        parse_jd_sheet(write_sheet(store, tmp_path / "s.txt", sheet), "d2-v1")
