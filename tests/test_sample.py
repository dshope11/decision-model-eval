"""Tests for decision_model_eval.sample."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from conftest import WriteJsonl, hit, run_header

from decision_model_eval.cli import main
from decision_model_eval.ingest import ingest_file
from decision_model_eval.jsonl import FormatError
from decision_model_eval.record import record_file
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

LAST_WEEK = "2026-09-26T20:00:00Z"
TONIGHT = "2026-10-03T20:00:00Z"


def ingest_at(conn: sqlite3.Connection, path: Path, ingested_at: str) -> None:
    result = ingest_file(conn, path)
    conn.execute("UPDATE runs SET ingested_at = ? WHERE run_uuid = ?",
                 (ingested_at, result.run_uuid))


@pytest.fixture
def tonight(conn: sqlite3.Connection, write_jsonl: WriteJsonl) -> sqlite3.Connection:
    """Last week: one listing. Tonight: ten listings, four of them filter rejects."""
    ingest_at(conn, write_jsonl(run_header(run_uuid="last-week"),
                                hit(source_key="old", title="Old Role")), LAST_WEEK)
    hits = [hit(source_key=str(k), title=f"Role {k}", passed_filter=k >= 4,
                filter_reason=None if k >= 4 else "not US or remote") for k in range(10)]
    ingest_at(conn, write_jsonl(run_header(run_uuid="tonight"), *hits), TONIGHT)
    return conn


def fill(sheet: Path, answers: list[str]) -> None:
    lines, it = [], iter(answers)
    for line in sheet.read_text().splitlines():
        if line.strip() == "D1:":
            line = f"    D1: {next(it)}"
        lines.append(line)
    sheet.write_text("\n".join(lines))


def make_sheet(conn: sqlite3.Connection, path: Path, size: int = 3, seed: int = 7) -> Path:
    items = draw(conn, TONIGHT, size, seed)
    write_sheet(items, path, name="blind-2026-W40", decision_point="D1",
                rubric_version="d1-v2", question="Worth reading the full JD? (y / n)",
                since=TONIGHT, seed=seed,
                population_size=len(population(conn, TONIGHT)))
    return path


def test_population_includes_rejects_and_excludes_earlier_runs(
    tonight: sqlite3.Connection,
) -> None:
    pool = population(tonight, TONIGHT)
    assert len(pool) == 10  # all ten tonight, rejects included; last week's listing not


def test_draw_is_reproducible_from_the_seed(tonight: sqlite3.Connection) -> None:
    first = draw(tonight, TONIGHT, 5, seed=42)
    assert first == draw(tonight, TONIGHT, 5, seed=42)
    assert len({item.listing_id for item in first}) == 5


def test_draw_larger_than_population_is_refused(tonight: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="exceeds the population"):
        draw(tonight, TONIGHT, 11, seed=42)


def test_draw_skips_excluded_listings(tonight: sqlite3.Connection) -> None:
    first = {item.listing_id for item in draw(tonight, TONIGHT, 6, seed=42)}
    second = draw(tonight, TONIGHT, 4, seed=42, exclude=frozenset(first))
    assert first.isdisjoint(item.listing_id for item in second)
    with pytest.raises(ValueError, match="exceeds the population of 4"):
        draw(tonight, TONIGHT, 5, seed=42, exclude=frozenset(first))


def test_recorded_sample_labels_are_already_sampled(
    tonight: sqlite3.Connection, tmp_path: Path, write_jsonl: WriteJsonl,
) -> None:
    sheet = make_sheet(tonight, tmp_path / "sheet.txt")
    assert already_sampled(tonight, "D1") == set()
    assert len(sheet_listing_ids(sheet)) == 3  # readable before it is answered
    fill(sheet, ["y", "n", "n"])
    meta, entries = parse_sheet(sheet)
    record_file(tonight, write_jsonl(*sheet_records(meta, entries, labeler="david",
                                                    rubric_version="d1-v1")))
    assert already_sampled(tonight, "D1") == sheet_listing_ids(sheet)
    assert already_sampled(tonight, "D2") == set()


def test_window_can_be_closed(tonight: sqlite3.Connection) -> None:
    assert len(population(tonight, LAST_WEEK)) == 11
    assert len(population(tonight, LAST_WEEK, TONIGHT)) == 1  # tonight's run left out


def test_sample_name_is_used_once_recorded(
    tonight: sqlite3.Connection, tmp_path: Path, write_jsonl: WriteJsonl,
) -> None:
    sheet = make_sheet(tonight, tmp_path / "sheet.txt")
    assert not sample_name_used(tonight, "blind-2026-W40")
    fill(sheet, ["y", "n", "n"])
    meta, entries = parse_sheet(sheet)
    record_file(tonight, write_jsonl(*sheet_records(meta, entries, labeler="david",
                                                    rubric_version="d1-v2")))
    assert sample_name_used(tonight, "blind-2026-W40")


def test_sheet_header_reads_sheets_only(tonight: sqlite3.Connection, tmp_path: Path) -> None:
    header = sheet_header(make_sheet(tonight, tmp_path / "sheet.txt"))
    assert (header["sample"], header["decision_point"]) == ("blind-2026-W40", "D1")
    other = tmp_path / "notes.txt"
    other.write_text("# a heading: not a sheet\nbody\n")
    assert sheet_header(other) == {}


def test_rubric_comes_from_the_header_and_cannot_be_contradicted() -> None:
    assert resolve_rubric({"rubric_version": "d1-v2"}, None) == "d1-v2"
    assert resolve_rubric({"rubric_version": "d1-v2"}, "d1-v2") == "d1-v2"
    assert resolve_rubric({}, "d1-v1") == "d1-v1"  # a sheet drawn before the header
    with pytest.raises(ValueError, match="contradicts"):
        resolve_rubric({"rubric_version": "d1-v2"}, "d1-v1")
    with pytest.raises(ValueError, match="pass --rubric-version"):
        resolve_rubric({}, None)


def test_cli_draws_skip_unrecorded_sibling_sheets_and_refuse_reused_names(
    tonight: sqlite3.Connection, tmp_path: Path,
) -> None:
    tonight.commit()
    config = tmp_path / "config.yaml"
    config.write_text(f"db_path: {tmp_path / 'store' / 'listings.db'}\n"
                      f"backup_dir: {tmp_path / 'backups'}\n")
    samples = tmp_path / "samples"
    samples.mkdir()

    def sample(name: str, out: str) -> int:
        return main(["--config", str(config), "sample", "--since", TONIGHT, "--name", name,
                     "--out", str(samples / out), "--size", "5"])

    assert sample("a", "a.txt") == 0
    assert sample("b", "b.txt") == 0  # "a" is not recorded, but it sits beside "b"
    a, b = sheet_listing_ids(samples / "a.txt"), sheet_listing_ids(samples / "b.txt")
    assert len(a) == len(b) == 5 and a.isdisjoint(b)
    assert "5 already sampled excluded" in (samples / "b.txt").read_text()
    assert sample("a", "a2.txt") == 1  # the name is taken by a sheet on disk
    assert not (samples / "a2.txt").exists()
    assert sample("c", "c.txt") == 1  # all 10 listings are on a sheet already


def test_sheet_shows_the_listing_line_and_nothing_else(
    tonight: sqlite3.Connection, tmp_path: Path,
) -> None:
    text = make_sheet(tonight, tmp_path / "sheet.txt").read_text()
    assert "WindBorne Systems :: Role" in text
    assert "    work: ? | level: ?" in text  # the view's details line
    assert "not US or remote" not in text  # no filter verdict on the sheet
    assert "# sample: blind-2026-W40" in text
    meta, _ = parse_sheet(fill_all(tmp_path / "sheet.txt"))
    assert meta["rubric_version"] == "d1-v2"


def fill_all(sheet: Path) -> Path:
    fill(sheet, ["n"] * sum(line.strip() == "D1:" for line in sheet.read_text().splitlines()))
    return sheet


def test_filled_sheet_records_blind_labels_once(
    tonight: sqlite3.Connection, tmp_path: Path, write_jsonl: WriteJsonl,
) -> None:
    sheet = make_sheet(tonight, tmp_path / "sheet.txt")
    fill(sheet, ["y", "n", "Y"])
    meta, entries = parse_sheet(sheet)
    assert [e.answer for e in entries] == [True, False, True]
    records = sheet_records(meta, entries, labeler="david", rubric_version="d1-v1")
    path = write_jsonl(*records)
    assert record_file(tonight, path).labels == 3
    assert record_file(tonight, path).already_recorded
    rows = tonight.execute(
        "SELECT value, sample, saw_model_output, labeler, rubric_version FROM labels"
    ).fetchall()
    assert sorted(json.loads(r["value"]) for r in rows) == [False, True, True]
    assert {(r["sample"], r["saw_model_output"], r["labeler"], r["rubric_version"])
            for r in rows} == {("blind-2026-W40", 0, "david", "d1-v1")}


def test_same_sheet_gets_the_same_batch_uuid(
    tonight: sqlite3.Connection, tmp_path: Path,
) -> None:
    sheet = make_sheet(tonight, tmp_path / "sheet.txt")
    fill(sheet, ["y", "y", "n"])
    meta, entries = parse_sheet(sheet)
    one = sheet_records(meta, entries, labeler="david", rubric_version=None)
    two = sheet_records(meta, entries, labeler="david", rubric_version=None)
    assert one[0]["batch_uuid"] == two[0]["batch_uuid"]


def test_unanswered_entry_is_an_error(tonight: sqlite3.Connection, tmp_path: Path) -> None:
    sheet = make_sheet(tonight, tmp_path / "sheet.txt")
    fill(sheet, ["y", "", "n"])
    with pytest.raises(FormatError, match=r"no answer for \[2\]"):
        parse_sheet(sheet)


def test_answer_other_than_y_or_n_is_an_error(
    tonight: sqlite3.Connection, tmp_path: Path,
) -> None:
    sheet = make_sheet(tonight, tmp_path / "sheet.txt")
    fill(sheet, ["y", "maybe", "n"])
    with pytest.raises(FormatError, match="must be y or n"):
        parse_sheet(sheet)


def test_unsure_marks_and_lines_below_the_answer_are_kept(
    tonight: sqlite3.Connection, tmp_path: Path,
) -> None:
    sheet = make_sheet(tonight, tmp_path / "sheet.txt")
    fill(sheet, ["y?", "n", "N ?"])
    text = sheet.read_text().replace(
        "    D1: n\n    note:", "    D1: n\n    note: location\n    explore: what is a GTM role", 1)
    sheet.write_text(text.replace("    D1: y?\n    note:", "    D1: y?\n    note:\n"
                                  "    explore on this role", 1))
    _, entries = parse_sheet(sheet)
    assert [(e.answer, e.note) for e in entries] == [
        (True, "unsure; explore on this role"),
        (False, "location; explore: what is a GTM role"),
        (False, "unsure"),
    ]
