"""Reading and validating the JSONL files that feed the store.

Every record is checked against an explicit key set: an unknown key is an error,
not silently dropped, so a typo in a fetch script fails loudly at ingest time.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1


class FormatError(ValueError):
    """A JSONL record that does not match the format."""


def read_records(path: str | Path) -> list[tuple[int, dict[str, Any]]]:
    """``(line_number, record)`` for every non-blank line of a JSONL file."""
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                raise FormatError(f"line {number}: not valid JSON ({err})") from err
            if not isinstance(record, dict):
                raise FormatError(f"line {number}: expected a JSON object")
            records.append((number, record))
    if not records:
        raise FormatError(f"{path}: empty file")
    return records


def check_keys(
    record: dict[str, Any], line: int, required: Iterable[str], optional: Iterable[str]
) -> None:
    """Raise unless ``record`` has every required key and nothing unexpected."""
    required, optional = set(required), set(optional)
    missing = required - record.keys()
    if missing:
        raise FormatError(f"line {line}: missing keys {sorted(missing)}")
    unknown = record.keys() - required - optional
    if unknown:
        raise FormatError(f"line {line}: unknown keys {sorted(unknown)}")


def timestamp(value: Any, line: int, key: str) -> str:
    """Normalize an ISO 8601 timestamp with a UTC offset to ``YYYY-MM-DDTHH:MM:SSZ``."""
    if not isinstance(value, str):
        raise FormatError(f"line {line}: {key} must be an ISO 8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as err:
        raise FormatError(f"line {line}: {key} is not ISO 8601: {value!r}") from err
    if parsed.tzinfo is None:
        raise FormatError(f"line {line}: {key} has no UTC offset: {value!r}")
    return parsed.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def date_or_timestamp(value: Any, line: int, key: str) -> str:
    """A bare date (``YYYY-MM-DD``) passes through; anything else must be a timestamp."""
    if isinstance(value, str) and len(value) == 10:
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError as err:
            raise FormatError(f"line {line}: {key} is not a date: {value!r}") from err
    return timestamp(value, line, key)


def boolean(value: Any, line: int, key: str) -> int:
    if not isinstance(value, bool):
        raise FormatError(f"line {line}: {key} must be true or false")
    return int(value)


def text(value: Any, line: int, key: str) -> str | None:
    """An optional string field."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise FormatError(f"line {line}: {key} must be a string")
    return value


def required_text(value: Any, line: int, key: str) -> str:
    """A required, non-blank string field."""
    if not isinstance(value, str) or not value.strip():
        raise FormatError(f"line {line}: {key} must be a non-empty string")
    return value


def number(value: Any, line: int, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise FormatError(f"line {line}: {key} must be a number")
    return float(value)


def integer(value: Any, line: int, key: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise FormatError(f"line {line}: {key} must be an integer")
    return int(value)


def canonical_json(value: Any) -> str:
    """Deterministic JSON text, so equal content always hashes equal."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
