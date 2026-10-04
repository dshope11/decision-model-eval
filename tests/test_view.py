"""Tests for decision_model_eval.view."""

from __future__ import annotations

import sqlite3
from typing import Any

from conftest import WriteJsonl, hit, run_header

from decision_model_eval.ingest import ingest_file
from decision_model_eval.view import ListingView, listing_view


def view_of(conn: sqlite3.Connection, write_jsonl: WriteJsonl, source: str,
            **fields: Any) -> ListingView:
    ingest_file(conn, write_jsonl(run_header(source=source), hit(**fields)))
    row = conn.execute("SELECT id FROM snapshots").fetchone()
    return listing_view(conn, row["id"])


def test_waas_view_cleans_location_and_reads_the_source_record(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl,
) -> None:
    job = {"remote": "only", "min_experience": 3, "job_type": "fulltime",
           "role": "eng", "eng_type": ["ml", "be"]}
    view = view_of(conn, write_jsonl, "waas", company="Quickchat AI",
                   title="Machine Learning Engineer",
                   location="PL; Masovian Voivodeship, PL; Warsaw, Masovian Voivodeship, PL",
                   raw={"job": job, "company": {}})
    assert view.line == ("Quickchat AI :: Machine Learning Engineer :: "
                         "Warsaw, Masovian Voivodeship, Poland")
    assert view.details == ("work: remote only", "level: 3+ yrs", "role: eng: ML, backend")


def test_waas_keeps_us_and_shows_interns(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl,
) -> None:
    job = {"remote": "no", "min_experience": 0, "job_type": "intern", "role": "marketing",
           "eng_type": ["fs"]}  # a stale subtype from another role is not shown
    view = view_of(conn, write_jsonl, "waas",
                   location="US; NY, US; New York, NY, US; New York; NY",
                   raw={"job": job, "company": {}})
    assert view.location == "New York, NY, US"
    assert view.details == ("work: on-site", "level: intern, 0+ yrs", "role: marketing")


def test_city_that_repeats_as_its_region_is_shown_once(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl,
) -> None:
    view = view_of(conn, write_jsonl, "waas", location="ID; Jakarta, ID; Jakarta, Jakarta, ID",
                   raw={"job": {}, "company": {}})
    assert view.location == "Jakarta, Jakarta, Indonesia"


def test_getro_view_uses_the_plain_location_list(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl,
) -> None:
    raw = {"id": 1, "work_mode": "on_site",
           "locations": ["San Francisco, CA, USA", "Oakland, CA, USA"]}
    view = view_of(conn, write_jsonl, "getro:103", level="mid_senior",
                   location="San Francisco, CA, USA; San Francisco County, CA, USA; "
                            "California, USA; United States; North America",
                   comp_min=150000, comp_max=200000, comp_currency="USD",
                   comp_period="year", raw=raw)
    assert view.location == "San Francisco, CA, USA; Oakland, CA, USA"
    assert view.details == ("work: on-site", "level: mid-senior", "pay: $150k-200k/yr")


def test_source_without_structured_fields_shows_unknowns(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl,
) -> None:
    # A two-letter state is not a country outside WaaS: CA stays CA.
    view = view_of(conn, write_jsonl, "linkedin-alert", location="San Francisco, CA",
                   comp_min=45, comp_currency="EUR", comp_period="hour", raw={"card": []})
    assert view.location == "San Francisco, CA"
    assert view.details == ("work: ?", "level: ?", "pay: EUR 45+/hr")


def test_missing_location_is_a_question_mark(
    conn: sqlite3.Connection, write_jsonl: WriteJsonl,
) -> None:
    view = view_of(conn, write_jsonl, "waas", location=None,
                   raw={"job": {"remote": "only"}, "company": {}})
    assert view.location == "?"
    assert view.details[0] == "work: remote only"
