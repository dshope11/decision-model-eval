"""Tests for decision_model_eval.ats."""

from __future__ import annotations

import pytest

from decision_model_eval.ats import AtsId, parse_ats


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://job-boards.greenhouse.io/anthropic/jobs/5198255008",
         AtsId("greenhouse", "anthropic", "5198255008")),
        ("https://boards.greenhouse.io/spacex/jobs/8559035002",
         AtsId("greenhouse", "spacex", "8559035002")),
        ("https://job-boards.eu.greenhouse.io/physicsx/jobs/4880947101",
         AtsId("greenhouse", "physicsx", "4880947101")),
        ("https://jobs.ashbyhq.com/openai/bba18df5-f30f-4d2c-909c-30e651f95579",
         AtsId("ashby", "openai", "bba18df5-f30f-4d2c-909c-30e651f95579")),
        ("https://jobs.lever.co/example/0f3c2f6e-1a2b-4c3d-9e8f-123456789abc/apply",
         AtsId("lever", "example", "0f3c2f6e-1a2b-4c3d-9e8f-123456789abc")),
        ("https://www.workatastartup.com/jobs/81736", AtsId("waas", None, "81736")),
        ("https://example.com/careers/role?gh_jid=7899565",
         AtsId("greenhouse", None, "7899565")),
    ],
)
def test_recognized_urls(url: str, expected: AtsId) -> None:
    assert parse_ats(url) == expected


@pytest.mark.parametrize(
    "url",
    [None, "", "https://acmerobotics.example/careers/software-engineer",
     "https://job-boards.greenhouse.io/anthropic"],
)
def test_unrecognized_urls(url: str | None) -> None:
    assert parse_ats(url) is None
