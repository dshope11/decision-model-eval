"""JD sheets: the D2 gates and the D5 tier, labeled on the full JD.

A JD sheet holds the listings a labeler marked D1 = y on a blind sample's D1
sheet, each with its verbatim JD, and is filled in once per JD. The header names
the sample, ``# decision_point: D2+D5`` and the ``# rubric_version:``. The rubric
version decides how the sheet is read: a rubric revision changes the gates, so a
sheet is only read under a version registered in ``RUBRICS``.

Each entry, after its JD block (everything up to ``----- end JD -----`` is the
sheet's own text and is skipped)::

    [4] L1 S1   (your D1: y)
        ...the listing view, the D1 note, the JD...
        G1 <name>: pass, fail or another verdict; ? for unsure; then any comment
        G2 <name>: pass (strict) | ext-fail: <what the rubric has no gate for>
        G3 <name>: key=value fields in place of a verdict
        G4 <name>: held=a, b | not=c  pass (1 of 3 not held)
        D5: high
        D1-check: y
        note: ...
        off-rubric: ...
        opened-link ...

Gate names are carried through and never checked: a rubric version says only how
many gates it has, which verdicts and answers it allows and which fields hold
lists. Each gate becomes a ``D2.G<n>`` label (the decision point ``decide()`` uses
for that gate) whose value is an object - ``gate``, ``verdict``, any fields, and
``unsure`` / ``pref_only`` / ``ext_fail`` when the line says so (``?``,
``pref-only``, ``| ext-fail: <text>``) - with the comment after the verdict as its
note. D5 and D1-check become labels whose value is the answer. The note,
off-rubric and any other line go to the D5 note, and an ``opened-link`` line to
the note of every label of its entry, since the whole entry was judged with the
live posting open.

A line carrying a ``[rev]`` marker was completed with a model's help after
labeling, then approved by the labeler. Its label records ``saw_model_output:
true``, so it never counts as a blind label; the marker (``[rev; was: ...]``) is
kept in its note.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from decision_model_eval.jsonl import FormatError
from decision_model_eval.sample import sheet_batch

JD_SHEET_POINT = "D2+D5"


@dataclass(frozen=True)
class JdRubric:
    gates: int
    verdicts: tuple[str, ...]
    list_fields: frozenset[str]
    tiers: tuple[str, ...]
    d1_checks: tuple[str, ...]


RUBRICS: dict[str, JdRubric] = {
    "d2-v1": JdRubric(gates=7, verdicts=("pass", "fail", "tier-drop"),
                      list_fields=frozenset({"held", "not"}),
                      tiers=("very-high", "high", "medium", "drop"),
                      d1_checks=("y", "n", "partly")),
}


@dataclass(frozen=True)
class JdLabel:
    decision_point: str
    value: Any
    note: str | None
    saw_model_output: bool


@dataclass(frozen=True)
class JdEntry:
    listing_id: int
    snapshot_id: int
    labels: tuple[JdLabel, ...]


_ENTRY = re.compile(r"^\[(\d+)\] L(\d+) S(\d+)\b")
_GATE = re.compile(r"^G(\d+)\s+([^:]+):(.*)$")
_REV = re.compile(r"\[rev\b[^\]]*\]")
_FIELD = re.compile(r"(?<![\w-])([A-Za-z][\w-]*)=")
_EXT_FAIL = re.compile(r"\|\s*ext-fail:\s*(.*)$")
_PREF_ONLY = re.compile(r"\bpref-only\b")
_JD_START, _JD_END = "----- JD, verbatim -----", "----- end JD -----"
_RESERVED = {"gate", "verdict", "unsure", "pref_only", "ext_fail"}


@dataclass
class _Answer:
    value: Any
    notes: list[str]
    rev: bool


@dataclass
class _Open:
    k: str
    listing_id: int
    snapshot_id: int
    answering: bool = False
    gates: dict[int, _Answer] = field(default_factory=dict)
    d5: _Answer | None = None
    d1_check: _Answer | None = None
    notes: list[str] = field(default_factory=list)
    opened: list[str] = field(default_factory=list)


def _without_rev(text: str) -> tuple[str, str | None]:
    """The answer text with its [rev] marker taken out, and the marker."""
    match = _REV.search(text)
    if match is None:
        return text.strip(), None
    rest = text[:match.start()].rstrip() + " " + text[match.end():].lstrip()
    return rest.strip(), match.group(0)


def _verdict_pattern(rubric: JdRubric) -> re.Pattern[str]:
    words = "|".join(re.escape(v) for v in sorted(rubric.verdicts, key=len, reverse=True))
    return re.compile(rf"(?:^|(?<=\s))({words})(\?)?(?=[\s,;:(]|$)")


def _verdict(text: str, pattern: re.Pattern[str], where: str,
             required: bool) -> tuple[dict[str, Any], str]:
    """The verdict object and the comment after it, for text that opens with a verdict."""
    match = pattern.match(text)
    if match is None:
        if required:
            raise FormatError(f"{where}: expected a verdict, got {text!r}")
        return {}, text.strip()
    value: dict[str, Any] = {"verdict": match.group(1)}
    if match.group(2):
        value["unsure"] = True
    rest = text[match.end():].strip()
    ext_fail = _EXT_FAIL.search(rest)
    if ext_fail:
        value["ext_fail"] = ext_fail.group(1).strip()
        rest = rest[:ext_fail.start()].strip()
    if _PREF_ONLY.search(rest):
        value["pref_only"] = True
    return value, rest.lstrip(" ,;:-").strip()


def _gate(name: str, text: str, rubric: JdRubric, where: str) -> tuple[dict[str, Any], str]:
    """(value, comment) for one gate answer, its [rev] marker already removed."""
    if not text:
        raise FormatError(f"{where}: no answer")
    pattern = _verdict_pattern(rubric)
    if not _FIELD.match(text):
        verdict, comment = _verdict(text, pattern, where, required=True)
        return {"gate": name, **verdict}, comment
    fields: dict[str, Any] = {}
    pos, trailing = 0, ""
    while (match := _FIELD.match(text, pos)) is not None:
        key = match.group(1)
        if key in fields or key in _RESERVED:
            raise FormatError(f"{where}: field {key!r} given twice or reserved")
        following = _FIELD.search(text, match.end())
        end = following.start() if following else len(text)
        region = text[match.end():end]
        verdict_at = pattern.search(region) if following is None else None
        if verdict_at is not None:
            trailing, region = region[verdict_at.start():], region[:verdict_at.start()]
        region = region.strip().strip("|").strip()
        if key in rubric.list_fields:
            fields[key] = [item.strip() for item in region.split(",") if item.strip()]
        elif not region:
            raise FormatError(f"{where}: field {key!r} has no value")
        elif following is not None:
            fields[key] = region  # every word up to the next field
        else:  # the last field: one word, then any comment
            token, _, rest = region.partition(" ")
            fields[key] = token
            trailing = " ".join(p for p in (rest.strip(), trailing) if p)
        if trailing:
            break
        pos = end
    verdict, comment = _verdict(trailing.strip(), pattern, where, required=False)
    return {"gate": name, **verdict, **fields}, comment


def _answer(text: str, allowed: tuple[str, ...], where: str) -> tuple[str, list[str]]:
    """(answer, notes) for a D5 or D1-check line: one allowed word, ? for unsure."""
    token, _, rest = text.partition(" ")
    unsure = token.endswith("?")
    value = token.rstrip("?").lower()
    if value not in allowed:
        raise FormatError(f"{where}: answer must be one of {' | '.join(allowed)}, got {token!r}")
    comment = rest.lstrip(" ,;:-").strip()
    return value, (["unsure"] if unsure else []) + ([comment] if comment else [])


def _labels(entry: _Open) -> tuple[JdLabel, ...]:
    def label(point: str, answer: _Answer, extra: list[str]) -> JdLabel:
        parts = answer.notes + extra + entry.opened
        return JdLabel(point, answer.value, "; ".join(p for p in parts if p) or None,
                       answer.rev)

    labels = [label(f"D2.G{n}", entry.gates[n], []) for n in sorted(entry.gates)]
    assert entry.d5 is not None
    labels.append(label("D5", entry.d5, entry.notes))
    if entry.d1_check is not None:
        labels.append(label("D1-check", entry.d1_check, []))
    return tuple(labels)


def parse_jd_sheet(path: Path, rubric_version: str) -> list[JdEntry]:
    """The filled entries of a JD sheet, read under ``rubric_version``. Every entry
    must answer every gate of that rubric and D5."""
    rubric = RUBRICS.get(rubric_version)
    if rubric is None:
        raise FormatError(f"{path}: no JD-sheet format for rubric_version "
                          f"{rubric_version!r} (known: {', '.join(sorted(RUBRICS))})")
    entries: list[JdEntry] = []
    missing: list[str] = []
    current: _Open | None = None
    in_jd = False

    def close() -> None:
        if current is None:
            return
        gaps = [f"G{n}" for n in range(1, rubric.gates + 1) if n not in current.gates]
        if current.d5 is None:
            gaps.append("D5")
        if gaps:
            missing.append(f"[{current.k}] {', '.join(gaps)}")
        else:
            entries.append(JdEntry(current.listing_id, current.snapshot_id, _labels(current)))

    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line, where = raw.strip(), f"{path}:{number}"
        if in_jd:
            if line == _JD_END:
                in_jd = False
                if current is not None:
                    current.answering = True
            continue
        if line == _JD_START:
            if current is None:
                raise FormatError(f"{where}: JD block outside an entry")
            in_jd = True
            continue
        if line.startswith("#"):
            continue
        match = _ENTRY.match(line)
        if match:
            close()
            current = _Open(match.group(1), int(match.group(2)), int(match.group(3)))
            continue
        if current is None or not line:
            continue
        gate = _GATE.match(line)
        if not current.answering:
            if not gate and not line.startswith("D5:"):
                continue  # the listing view and D1 note above an entry's answers
            current.answering = True
        if gate:
            n = int(gate.group(1))
            if not 1 <= n <= rubric.gates:
                raise FormatError(f"{where}: {rubric_version} has gates G1-G{rubric.gates}")
            if n in current.gates:
                raise FormatError(f"{where}: G{n} answered twice in [{current.k}]")
            text, rev = _without_rev(gate.group(3))
            value, comment = _gate(gate.group(2).strip(), text, rubric, where)
            current.gates[n] = _Answer(value, [comment, rev or ""], rev is not None)
        elif line.startswith("D5:") or line.startswith("D1-check:"):
            is_d5 = line.startswith("D5:")
            if (current.d5 if is_d5 else current.d1_check) is not None:
                raise FormatError(f"{where}: answered twice in [{current.k}]")
            text, rev = _without_rev(line.partition(":")[2])
            word, notes = _answer(text, rubric.tiers if is_d5 else rubric.d1_checks, where)
            answer = _Answer(word, notes + [rev or ""], rev is not None)
            if is_d5:
                current.d5 = answer
            else:
                current.d1_check = answer
        elif line.startswith("note:"):
            current.notes.append(line[len("note:"):].strip())
        elif line.startswith("opened-link"):
            current.opened.append(line)
        else:
            current.notes.append(line)  # off-rubric and anything else, verbatim
    if in_jd:
        raise FormatError(f"{path}: JD block not closed with {_JD_END!r}")
    close()
    if missing:
        raise FormatError(f"{path}: no answer for {'; '.join(missing)}")
    return entries


def jd_sheet_records(meta: dict[str, str], entries: list[JdEntry], *, labeler: str,
                     rubric_version: str) -> list[dict[str, Any]]:
    """A ``dme record`` annotations file for a filled JD sheet. A label records
    ``saw_model_output: true`` only for a [rev] line."""
    name = meta["sample"]
    records: list[dict[str, Any]] = [sheet_batch(name, meta["decision_point"], labeler)]
    for entry in entries:
        for label in entry.labels:
            record: dict[str, Any] = {
                "type": "label", "listing": {"listing_id": entry.listing_id},
                "snapshot_id": entry.snapshot_id, "decision_point": label.decision_point,
                "labeler": labeler, "value": label.value,
                "saw_model_output": label.saw_model_output, "sample": name,
                "rubric_version": rubric_version,
            }
            if label.note:
                record["note"] = label.note
            records.append(record)
    return records
