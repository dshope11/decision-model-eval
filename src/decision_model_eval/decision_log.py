"""The decision log: one JSONL line per backend call, written as the call returns.

It is format 2 of the annotations format ``dme record`` reads (``record.py``). Line 1
is the run header:

    {"type": "batch", "format": 2, "batch_uuid": str, "origin": str, "run": object}

``batch_uuid`` is the run's UUID, so recording a log twice is a no-op. ``run`` holds the
run configuration as the caller supplies it (models with effort, repeats, seed, code
commit, dataset version). Every later line is one attempt at one decision; its fields are
listed in ``client.py``. Failed attempts are logged too, with ``error`` set and ``output``
null, so failure rates can be counted; ``dme record`` skips them.

Each line is flushed and fsynced before the call returns, so a crash loses at most the
call in flight, never one that was paid for. A log is never reopened: a resumed run
writes a new file.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from types import TracebackType
from typing import Any

LOG_FORMAT = 2


class DecisionLog:
    """Append-only writer for one run's decision log. Safe to share between threads."""

    def __init__(self, path: str | Path, *, origin: str, run: dict[str, Any],
                 run_uuid: str | None = None) -> None:
        self.path = Path(path)
        self.run_uuid = run_uuid or str(uuid.uuid4())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._file = self.path.open("x", encoding="utf-8")  # refuses an existing file
        self.write({"type": "batch", "format": LOG_FORMAT, "batch_uuid": self.run_uuid,
                    "origin": origin, "run": run})

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            self._file.write(line)
            self._file.flush()
            os.fsync(self._file.fileno())

    def close(self) -> None:
        with self._lock:
            self._file.close()

    def __enter__(self) -> DecisionLog:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.close()
