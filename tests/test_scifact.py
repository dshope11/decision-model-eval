"""Tests for decision_model_eval.scifact."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest
from conftest import JEV_CHOICE, haiku_response

from decision_model_eval import scifact
from decision_model_eval.cli import main
from decision_model_eval.decision_log import DecisionLog
from decision_model_eval.jsonl import FormatError
from decision_model_eval.questions import question_hash

CORPUS = [
    {"doc_id": 10, "title": "Aspirin trial", "structured": False,
     "abstract": ["Aspirin reduced incidence.", "It was well tolerated."]},
    {"doc_id": 20, "title": "Cohort study", "structured": True,
     "abstract": ["OBJECTIVE To test it.   \n", "RESULTS No effect.   \n"]},
    {"doc_id": 30, "title": "Unrelated", "structured": False, "abstract": ["Nothing here."]},
]
CLAIMS = [
    {"id": 1, "claim": "Aspirin reduces incidence.", "cited_doc_ids": [10, 20],
     "evidence": {"10": [{"sentences": [0], "label": "SUPPORT"},
                         {"sentences": [1], "label": "SUPPORT"}],
                  "20": [{"sentences": [1], "label": "CONTRADICT"}]}},
    {"id": 2, "claim": "Aspirin causes harm.", "cited_doc_ids": [30, 30], "evidence": {}},
]


def _jsonl(records: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r) + "\n" for r in records).encode("utf-8")


def _tarball(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(f"data/{name}")
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pin(monkeypatch: pytest.MonkeyPatch, claims: list[dict[str, Any]]) -> bytes:
    """Pin the fixture release in place of the real one; return its tarball."""
    files = {scifact.CLAIMS_FILE: _jsonl(claims), scifact.CORPUS_FILE: _jsonl(CORPUS)}
    tarball = _tarball(files)
    monkeypatch.setattr(scifact, "FILES", {n: _sha256(c) for n, c in files.items()})
    monkeypatch.setattr(scifact, "RELEASE_SHA256", _sha256(tarball))
    return tarball


@pytest.fixture
def release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    scifact.download(tmp_path, fetch=lambda _: _pin(monkeypatch, CLAIMS))
    return tmp_path


def test_pairs_labels_and_order(release: Path) -> None:
    pairs = scifact.load_pairs(release)
    assert [(p.claim_id, p.doc_id, p.label) for p in pairs] == [
        (1, 10, "supports"), (1, 20, "contradicts"), (2, 30, "not_enough_info")]
    assert [p.supports_label for p in pairs] == [
        "supports", "does_not_support", "does_not_support"]


def test_state_and_item(release: Path) -> None:
    pair = scifact.load_pairs(release)[1]
    assert pair.state == {"claim": "Aspirin reduces incidence.", "title": "Cohort study",
                          "abstract": "OBJECTIVE To test it. RESULTS No effect."}
    assert pair.item == {"dataset": "scifact", "split": "dev", "claim_id": 1, "doc_id": 20}


def test_labels_are_options_of_the_questions(release: Path) -> None:
    for pair in scifact.load_pairs(release):
        assert pair.label in scifact.VERDICT.options
        assert pair.supports_label in scifact.SUPPORTS.options
    assert scifact.VERDICT.instructions == scifact.SUPPORTS.instructions
    assert question_hash(scifact.VERDICT) != question_hash(scifact.SUPPORTS)


def test_download_is_a_no_op_once_verified(release: Path) -> None:
    def fail(_: str) -> bytes:
        raise AssertionError("fetched again")
    assert scifact.download(release, fetch=fail) is False


def test_changed_file_is_refused(release: Path) -> None:
    (release / scifact.CORPUS_FILE).write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="dme scifact download"):
        scifact.load_pairs(release)


def test_wrong_release_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tarball = _pin(monkeypatch, CLAIMS)
    monkeypatch.setattr(scifact, "RELEASE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="pinned"):
        scifact.download(tmp_path / "out", fetch=lambda _: tarball)
    assert not (tmp_path / "out").exists()


def test_changed_member_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tarball = _pin(monkeypatch, CLAIMS)
    monkeypatch.setitem(scifact.FILES, scifact.CORPUS_FILE, "0" * 64)
    with pytest.raises(ValueError, match=r"data/corpus\.jsonl"):
        scifact.download(tmp_path / "out", fetch=lambda _: tarball)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(("claim", "message"), [
    ({"cited_doc_ids": [10], "evidence": {"20": [{"sentences": [0], "label": "SUPPORT"}]}},
     "does not cite"),
    ({"cited_doc_ids": [99], "evidence": {}}, "not in the corpus"),
    ({"cited_doc_ids": [10], "evidence": {"10": [{"sentences": [0], "label": "SUPPORT"},
                                                 {"sentences": [1], "label": "CONTRADICT"}]}},
     "labels"),
    ({"cited_doc_ids": [10], "evidence": {"10": [{"sentences": [0], "label": "NEI"}]}},
     "labels"),
])
def test_bad_claims_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                 claim: dict[str, Any], message: str) -> None:
    claims = [{"id": 7, "claim": "A claim.", **claim}]
    scifact.download(tmp_path, fetch=lambda _: _pin(monkeypatch, claims))
    with pytest.raises(FormatError, match=message):
        scifact.load_pairs(tmp_path)


def test_question_wording_is_frozen() -> None:
    # Approved 2026-10-20. A run's logs pair backends on these hashes, so a change in
    # wording is a new question and a new run, never an edit.
    assert {point: question_hash(q) for point, q in scifact.POINTS.items()} == {
        "scifact.verdict": "3a97260b9dafc93a", "scifact.supports": "fe4eebedc0739cb4"}


def _respond(fail_claim: str | None = None) -> Any:
    """A stateless fake transport: the first option, at probability 1, from either API."""
    def respond(url: str, body: dict[str, Any], headers: dict[str, str],
                timeout: float) -> tuple[int, Any]:
        if url.endswith("/alpha/decisions"):
            if body["state"]["claim"] == fail_claim:
                return 400, {"error": {"message": "bad request"}}
            options = list(body["questions"]["question"]["criteria"])
        else:
            options = body["output_config"]["format"]["schema"]["properties"]["answer"]["enum"]
        probabilities = {option: 0.0 for option in options} | {options[0]: 1.0}
        if url.endswith("/alpha/decisions"):
            return 200, {**JEV_CHOICE, "answers": {"question": {
                "type": "choice", "choice": options[0], "probabilities": probabilities,
                "confidence": 1.0}}}
        return 200, haiku_response(json.dumps({"answer": options[0],
                                               "probabilities": probabilities}))
    return respond


BACKENDS = ["jev", "haiku"]


def test_plan(release: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scifact, "SUPPORTS_SUBSET", 2)
    pairs = scifact.load_pairs(release)
    calls = scifact.plan(pairs, backends=BACKENDS, seed=42)
    assert len(calls) == (3 * 2 + 2 * 1) * scifact.REPEATS
    assert len({call.key for call in calls}) == len(calls)
    assert {c.backend for c in calls if c.point == "scifact.supports"} == {"jev"}
    smoke = scifact.plan(pairs, backends=BACKENDS, seed=42, limit=1)
    assert len(smoke) == (1 * 2 + 1 * 1) * scifact.REPEATS
    assert scifact.supports_subset(pairs, 42) == scifact.supports_subset(pairs, 42)


@pytest.mark.usefixtures("api_key")
def test_run_resumes_only_unanswered_calls(release: Path, tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scifact, "SUPPORTS_SUBSET", 2)
    calls = scifact.plan(scifact.load_pairs(release), backends=BACKENDS, seed=42)
    settings = scifact.run_config(backends=BACKENDS, seed=42, limit=None)
    run_dir = tmp_path / "run"
    broken = [c for c in calls if c.backend == "jev" and c.pair.claim_id == 2]
    with DecisionLog(run_dir / "log-1.jsonl", origin="test", run=settings) as log:
        assert scifact.run(calls, log=log, workers=4,
                           transport=_respond(fail_claim="Aspirin causes harm.")) == (
            len(calls) - len(broken), len(broken))
    done = scifact.answered(run_dir, settings)
    assert [c for c in calls if c.key not in done] == broken
    with DecisionLog(run_dir / "log-2.jsonl", origin="test",
                     run={**settings, "commit": "abc123"}) as log:
        assert scifact.run(broken, log=log, workers=4, transport=_respond()) == (
            len(broken), 0)
    assert scifact.answered(run_dir, settings) == {c.key for c in calls}


def test_a_log_of_another_run_is_refused(tmp_path: Path) -> None:
    settings = scifact.run_config(backends=BACKENDS, seed=42, limit=None)
    with DecisionLog(tmp_path / "log-1.jsonl", origin="test", run=settings):
        pass
    with pytest.raises(ValueError, match="another run configuration"):
        scifact.answered(tmp_path, {**settings, "limit": 10})


def test_cli_download(release: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(f"scifact_dir: {release}\n")
    assert main(["--config", str(config), "scifact", "download"]) == 0
    assert capsys.readouterr().out == (
        f"{release}: already present; 3 dev pairs "
        "(1 supports, 1 contradicts, 1 not_enough_info)\n")
