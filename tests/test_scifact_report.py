"""Tests for decision_model_eval.scifact_report, on a synthetic run with known answers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import JEV_CHOICE, haiku_response
from test_scifact import _jsonl, _sha256, _tarball

from decision_model_eval import scifact, scifact_report
from decision_model_eval.decision_log import DecisionLog

EVIDENCE = {"supports": "SUPPORT", "contradicts": "CONTRADICT"}
LABELS = scifact.VERDICT.options
CLAIMS = [{"id": i, "claim": f"Claim {i} [{LABELS[i % 3]}]", "cited_doc_ids": [100 + i],
           "evidence": ({str(100 + i): [{"sentences": [0], "label": EVIDENCE[LABELS[i % 3]]}]}
                        if LABELS[i % 3] in EVIDENCE else {})}
          for i in range(12)]
CORPUS = [{"doc_id": 100 + i, "title": f"Doc {i}", "structured": False,
           "abstract": ["A sentence."]} for i in range(12)]
BACKENDS = ["jev", "haiku"]


def _answer(body: dict[str, Any], options: list[str]) -> tuple[str, dict[str, float]]:
    """Right at 0.8 on even claims, wrong at 0.6 on odd ones."""
    found = re.search(r"Claim (\d+) \[(\w+)\]", json.dumps(body))
    assert found is not None
    number, label = int(found.group(1)), found.group(2)
    true = label if label in options else "does_not_support"
    answer = true if number % 2 == 0 else next(o for o in options if o != true)
    top = 0.8 if number % 2 == 0 else 0.6
    rest = (1 - top) / (len(options) - 1)
    return answer, {o: top if o == answer else rest for o in options}


def _respond(url: str, body: dict[str, Any], headers: dict[str, str],
             timeout: float) -> tuple[int, Any]:
    if url.endswith("/alpha/decisions"):
        options = list(body["questions"]["question"]["criteria"])
        answer, probabilities = _answer(body, options)
        k = len(options)
        return 200, {**JEV_CHOICE, "answers": {"question": {
            "type": "choice", "choice": answer, "probabilities": probabilities,
            "confidence": (k * probabilities[answer] - 1) / (k - 1)}}}
    options = body["output_config"]["format"]["schema"]["properties"]["answer"]["enum"]
    answer, probabilities = _answer(body, options)
    return 200, haiku_response(json.dumps({"answer": answer, "probabilities": probabilities}))


@pytest.fixture
def run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    files = {scifact.CLAIMS_FILE: _jsonl(CLAIMS), scifact.CORPUS_FILE: _jsonl(CORPUS)}
    tarball = _tarball(files)
    monkeypatch.setattr(scifact, "FILES", {n: _sha256(c) for n, c in files.items()})
    monkeypatch.setattr(scifact, "RELEASE_SHA256", _sha256(tarball))
    monkeypatch.setattr(scifact, "SUPPORTS_SUBSET", 6)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    scifact.download(tmp_path, fetch=lambda _: tarball)
    run = tmp_path / "runs" / "synthetic"
    calls = scifact.plan(scifact.load_pairs(tmp_path), backends=BACKENDS, seed=42)
    settings = scifact.run_config(backends=BACKENDS, seed=42, limit=None)
    with DecisionLog(run / "log-1.jsonl", origin="test", run=settings) as log:
        assert scifact.run(calls, log=log, workers=4, transport=_respond) == (len(calls), 0)
    return run


def test_report(run_dir: Path) -> None:
    results, text, written = scifact_report.report(run_dir, run_dir.parents[1])
    assert [p.name for p in written] == [
        "results.json", "summary.md", "items.jsonl", "top_probability.png",
        "reliability_mean.png", "reliability_vote.png"]
    assert json.loads((run_dir / "report" / "results.json").read_text()) == results
    rows = [json.loads(line) for line in (run_dir / "report" / "items.jsonl").open()]
    assert len(rows) == 12 + 6
    assert rows[0]["jev"]["answer"] == rows[0]["label"] == "supports"  # claim 0, even
    verdict = results["points"]["scifact.verdict"]
    for entry in verdict.values():
        assert entry["accuracy"]["k"] == 6 and entry["accuracy"]["n"] == 12
        assert entry["answer_changed_across_repeats"] == 0.0
        # Top option: 6 items at 0.8, all right; 6 at 0.6, all wrong -> |1-0.8|, |0-0.6|
        assert entry["calibration"]["mean"]["top_label_ece"] == pytest.approx(0.4)
        floor = entry["calibration"]["mean"]["top_label_ece_floor"]
        assert 0 < floor["mean"] < floor["p95"] < 0.4 and floor["seed"] == 42
    assert results["majority_baseline"]["accuracy"]["k"] == 4
    assert results["agreement"]["agreement"] == 1.0
    assert np.trace(np.array(results["agreement"]["matrix"])) == 12
    checks = [entry["confidence_vs_rescaled_top"] for point in results["points"].values()
              for entry in point.values() if "confidence_vs_rescaled_top" in entry]
    assert [c["K"] for c in checks] == [3, 2]
    assert all(c["max_abs_residual"] == pytest.approx(0.0) for c in checks)
    assert list(results["points"]["scifact.supports"]) == ["jev"]
    assert "## Working points" in text and "| jev | 9 | stated | 0.900 |" in text
    assert "(95th pct " in text and "ECE floor: the top-label ECE" in text


def test_stated_rows_act_at_break_even(run_dir: Path) -> None:
    results, _, _ = scifact_report.report(run_dir, run_dir.parents[1])
    rows = results["points"]["scifact.verdict"]["jev"]["working_points"]["mean"]
    # c = 1 (0.5) answers all; c = 3 (0.75) only the 0.8s, all right; c = 9 (0.9) none
    assert [rows[f"c={c}"]["stated"]["threshold"] for c in (1, 3, 9)] == [0.5, 0.75, 0.9]
    assert rows["c=1"]["stated"]["coverage"] == 1.0
    assert rows["c=3"]["stated"]["wrong"] == 0
    assert rows["c=3"]["stated"]["accuracy_answered"] == 1.0
    assert rows["c=9"]["stated"]["answered"] == 0
    reporting = results["split"]["reporting"]
    assert sum(reporting.values()) == 6 and set(reporting.values()) == {2}


def test_split_is_seeded_and_stratified(run_dir: Path) -> None:
    pairs = scifact.load_pairs(run_dir.parents[1])
    first = scifact_report.split(pairs, 42)
    assert np.array_equal(first, scifact_report.split(pairs, 42))
    for label in LABELS:
        assert sum(s for p, s in zip(pairs, first, strict=True) if p.label == label) == 2


def test_stated_threshold_is_inclusive_despite_rounding() -> None:
    # The third score may land a rounding error below 0.9.
    score = np.array([0.8, 0.6, np.mean([0.9] * 5), 0.95, 0.5])
    rows = scifact_report.working_points(score, np.array([True, False, True, True, False]),
                                         np.array([True, True, False, False, False]))
    assert rows["c=9"]["stated"]["answered"] == 2


def test_an_unfinished_run_is_refused(run_dir: Path) -> None:
    log = run_dir / "log-1.jsonl"
    lines = log.read_text().splitlines()
    log.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="1 of .* planned calls unanswered"):
        scifact_report.report(run_dir, run_dir.parents[1])
