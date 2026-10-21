"""SciFact claim verification: the public control set (Wadden et al., EMNLP 2020).

The release is fetched by ``dme scifact download``, never shipped in the repo, and pinned
by SHA-256: the tarball when it is downloaded, and each extracted file every time it is
read. Claims are CC BY 4.0; the abstracts are ODC-By 1.0.

One item is a (claim, abstract) pair: each abstract a dev claim cites, labeled
``supports`` or ``contradicts`` where the claim's evidence names the abstract, and
``not_enough_info`` where it does not (a cited abstract with no rationale). The dev split
has 300 claims and 339 distinct pairs: 138 supports, 71 contradicts, 130 not_enough_info.
Claim 1245 lists one abstract twice; that is one pair.

Two questions are asked, both choices so Jev reports a ``confidence``: ``VERDICT`` (three
options, both backends) and ``SUPPORTS`` (two options, Jev only, on a seeded subset),
which tests whether Jev's confidence is the same rescaling of its top probability with
two options as with three.

A run is a directory of decision logs, ``<scifact_dir>/runs/<name>/log-<time>.jsonl``,
each headed by the same run configuration. A log is never reopened, so resuming an
interrupted run writes a new log that skips every call already answered.
"""

from __future__ import annotations

import hashlib
import io
import random
import sys
import tarfile
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decision_model_eval import jsonl
from decision_model_eval.backends import BACKENDS
from decision_model_eval.client import DecisionError, decide
from decision_model_eval.decision_log import DecisionLog
from decision_model_eval.jsonl import FormatError
from decision_model_eval.questions import Question, question_hash

RELEASE_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
# The tarball served on 2026-10-20 (Last-Modified 2021-01-26). The URL says "latest", so a
# new release would change this hash and fail the download rather than change the data.
RELEASE_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
SPLIT = "dev"
CLAIMS_FILE = f"claims_{SPLIT}.jsonl"
CORPUS_FILE = "corpus.jsonl"
# file name -> SHA-256 of its content; each is the tarball member data/<name>
FILES = {
    CLAIMS_FILE: "86f0435d08fdb65d1aa41d1472684f57e6e71930626497bdf4d7a9ec1a632217",
    CORPUS_FILE: "b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62",
}
EVIDENCE_LABELS = {"SUPPORT": "supports", "CONTRADICT": "contradicts"}
NO_EVIDENCE = "not_enough_info"

INSTRUCTIONS = ("Judge from the abstract alone, not from what you know about the topic: "
                "does the abstract give evidence that the claim is true or false?")
VERDICT = Question(type="choice", instructions=INSTRUCTIONS, criteria={
    "supports": "The abstract gives evidence that the claim is true.",
    "contradicts": "The abstract gives evidence that the claim is false.",
    NO_EVIDENCE: "The abstract gives no evidence either way.",
})
SUPPORTS = Question(type="choice", instructions=INSTRUCTIONS, criteria={
    "supports": "The abstract gives evidence that the claim is true.",
    "does_not_support": "The abstract gives evidence that the claim is false, or no "
                        "evidence either way.",
})
POINTS = {"scifact.verdict": VERDICT, "scifact.supports": SUPPORTS}
SUPPORTS_BACKEND = "jev"  # SUPPORTS tests Jev's confidence; Haiku reports none
SUPPORTS_SUBSET = 100
REPEATS = 5

Fetch = Callable[[str], bytes]


@dataclass(frozen=True)
class Pair:
    """One (claim, abstract) pair and its label, an option of ``VERDICT``."""

    claim_id: int
    doc_id: int
    claim: str
    title: str
    abstract: tuple[str, ...]  # sentences, as released
    label: str

    @property
    def item(self) -> dict[str, Any]:
        return {"dataset": "scifact", "split": SPLIT, "claim_id": self.claim_id,
                "doc_id": self.doc_id}

    @property
    def state(self) -> dict[str, Any]:
        # Structured abstracts carry a trailing "   \n" on every sentence.
        return {"claim": self.claim, "title": self.title,
                "abstract": " ".join(s.strip() for s in self.abstract)}

    @property
    def supports_label(self) -> str:
        """The label as an option of ``SUPPORTS``."""
        return "supports" if self.label == "supports" else "does_not_support"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _urllib_fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return bytes(response.read())


def _verified(directory: Path) -> bool:
    return all((directory / name).is_file()
               and _sha256((directory / name).read_bytes()) == digest
               for name, digest in FILES.items())


def download(directory: str | Path, *, fetch: Fetch = _urllib_fetch) -> bool:
    """Fetch the release and write the pinned files into ``directory``.

    Returns False, fetching nothing, when the files are already there with the right
    hashes. A tarball or file with another hash raises before anything is written.
    """
    directory = Path(directory).expanduser()
    if _verified(directory):
        return False
    data = fetch(RELEASE_URL)
    if _sha256(data) != RELEASE_SHA256:
        raise ValueError(f"{RELEASE_URL}: SHA-256 {_sha256(data)}, pinned {RELEASE_SHA256}")
    contents = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for name, digest in FILES.items():
            member = tar.extractfile(f"data/{name}")
            if member is None:
                raise ValueError(f"{RELEASE_URL}: data/{name} is not a file")
            contents[name] = member.read()
            if _sha256(contents[name]) != digest:
                raise ValueError(f"data/{name}: SHA-256 {_sha256(contents[name])}, "
                                 f"pinned {digest}")
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        (directory / name).write_bytes(content)
    return True


def load_pairs(directory: str | Path) -> list[Pair]:
    """Every distinct (claim, abstract) pair of the split, in release order."""
    directory = Path(directory).expanduser()
    if not _verified(directory):
        raise ValueError(f"{directory}: SciFact files missing or changed; run "
                         "`dme scifact download`")
    corpus = {}
    for line, doc in jsonl.read_records(directory / CORPUS_FILE):
        jsonl.check_keys(doc, line, ("doc_id", "title", "abstract", "structured"), ())
        corpus[doc["doc_id"]] = doc
    pairs = []
    for line, claim in jsonl.read_records(directory / CLAIMS_FILE):
        jsonl.check_keys(claim, line, ("id", "claim", "evidence", "cited_doc_ids"), ())
        cited = list(dict.fromkeys(claim["cited_doc_ids"]))
        labels = {int(doc_id): {rationale["label"] for rationale in rationales}
                  for doc_id, rationales in claim["evidence"].items()}
        if not labels.keys() <= set(cited):
            raise FormatError(f"line {line}: evidence for an abstract the claim does not cite")
        for doc_id in cited:
            if doc_id not in corpus:
                raise FormatError(f"line {line}: abstract {doc_id} is not in the corpus")
            found = labels.get(doc_id)
            if found is None:
                label = NO_EVIDENCE
            elif len(found) == 1 and next(iter(found)) in EVIDENCE_LABELS:
                label = EVIDENCE_LABELS[found.pop()]
            else:
                raise FormatError(f"line {line}: abstract {doc_id} has labels {sorted(found)}")
            doc = corpus[doc_id]
            pairs.append(Pair(claim_id=claim["id"], doc_id=doc_id, claim=claim["claim"],
                              title=doc["title"], abstract=tuple(doc["abstract"]),
                              label=label))
    return pairs


@dataclass(frozen=True)
class Call:
    """One planned decision: a question about a pair, asked of one backend once."""

    point: str
    backend: str
    pair: Pair
    repeat: int

    @property
    def key(self) -> str:
        return call_key(self.point, self.backend, self.pair.item, self.repeat)


def call_key(point: str, backend: str, item: dict[str, Any], repeat: int) -> str:
    return jsonl.canonical_json([point, backend, item, repeat])


def supports_subset(pairs: Sequence[Pair], seed: int) -> list[Pair]:
    """The seeded pairs ``SUPPORTS`` is asked about, in release order."""
    chosen = set(random.Random(seed).sample(range(len(pairs)), SUPPORTS_SUBSET))
    return [pair for i, pair in enumerate(pairs) if i in chosen]


def plan(pairs: Sequence[Pair], *, backends: Sequence[str], seed: int,
         limit: int | None = None) -> list[Call]:
    """Every call of a run, each pair's calls together. ``limit`` keeps the first pairs
    of each question's list, for a smoke run."""
    asked = [("scifact.verdict", list(pairs), list(backends)),
             ("scifact.supports", supports_subset(pairs, seed), [SUPPORTS_BACKEND])]
    return [Call(point, backend, pair, repeat)
            for point, items, keys in asked
            for pair in items[:limit]
            for backend in keys
            for repeat in range(REPEATS)]


def run_config(*, backends: Sequence[str], seed: int, limit: int | None) -> dict[str, Any]:
    """What a run's logs must agree on to be one run."""
    def models(keys: Sequence[str]) -> dict[str, str]:
        return {key: BACKENDS[key].model_label for key in keys}
    return {
        "dataset": "scifact", "split": SPLIT, "release_url": RELEASE_URL,
        "release_sha256": RELEASE_SHA256, "files": dict(FILES), "seed": seed,
        "repeats": REPEATS, "limit": limit, "supports_subset": SUPPORTS_SUBSET,
        "points": {
            "scifact.verdict": {"question_hash": question_hash(VERDICT),
                                "backends": models(backends)},
            "scifact.supports": {"question_hash": question_hash(SUPPORTS),
                                 "backends": models([SUPPORTS_BACKEND])},
        },
        "sampling": "each model's default; temperature and top_p are never sent",
    }


def run_logs(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("log-*.jsonl"))


def answered(run_dir: Path, config: dict[str, Any]) -> set[str]:
    """Keys of the calls a run's logs already answered. Refuses a log of another run."""
    done = set()
    for path in run_logs(run_dir):
        records = jsonl.read_records(path)
        logged = records[0][1].get("run", {})
        if {k: v for k, v in logged.items() if k != "commit"} != config:
            raise ValueError(f"{path}: logged with another run configuration; "
                             "start a run under a new name")
        for _, record in records[1:]:
            if record.get("type") == "decision" and record.get("error") is None:
                done.add(call_key(record["decision_point"], record["backend"],
                                  record["item"], record["repeat_index"]))
    return done


def run(calls: Sequence[Call], *, log: DecisionLog, workers: int,
        **decide_options: Any) -> tuple[int, int]:
    """Make the calls on ``workers`` threads; return (answered, failed).

    A failed call is in the log with its error and is retried by the next resume. On an
    interrupt, calls not yet started are dropped; calls in flight finish and are logged.
    """
    def ask(call: Call) -> None:
        decide(call.point, call.pair.state, POINTS[call.point], backend=call.backend,
               item=call.pair.item, log=log, repeat_index=call.repeat, **decide_options)

    ok = failed = 0
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        for future in as_completed([pool.submit(ask, call) for call in calls]):
            try:
                future.result()
                ok += 1
            except DecisionError as err:
                failed += 1
                print(f"failed: {err}", file=sys.stderr)
            if (ok + failed) % 100 == 0:
                print(f"{ok + failed} / {len(calls)}", file=sys.stderr)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return ok, failed
