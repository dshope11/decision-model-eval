"""Parse an applicant-tracking-system (ATS) identity out of a posting URL.

The same job is often seen through several sources - an alert, a VC board, the
company's own ATS. The ATS identity ``(ats, tenant, job_id)`` is what later lets
those listings be grouped as one posting. Parsing is deterministic and the URL
is always stored, so a pattern added here later can be back-applied.
"""

from __future__ import annotations

import re
from typing import NamedTuple
from urllib.parse import parse_qs, urlsplit


class AtsId(NamedTuple):
    ats: str
    tenant: str | None
    job_id: str


_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    # (ats, host pattern, path pattern with tenant and job_id groups)
    ("greenhouse", re.compile(r"^(job-boards(\.eu)?|boards)\.greenhouse\.io$"),
     r"^/(?P<tenant>[^/]+)/jobs/(?P<job_id>\d+)"),
    ("ashby", re.compile(r"^jobs\.ashbyhq\.com$"),
     r"^/(?P<tenant>[^/]+)/(?P<job_id>[0-9a-f-]{36})"),
    ("lever", re.compile(r"^jobs(\.eu)?\.lever\.co$"),
     r"^/(?P<tenant>[^/]+)/(?P<job_id>[0-9a-f-]{36})"),
    ("waas", re.compile(r"^(www\.)?workatastartup\.com$"),
     r"^/jobs/(?P<job_id>\d+)"),
]


def parse_ats(url: str | None) -> AtsId | None:
    """``AtsId`` for a recognized posting URL, else ``None``."""
    if not url:
        return None
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    for ats, host_pattern, path_pattern in _PATTERNS:
        if host_pattern.match(host):
            match = re.match(path_pattern, parts.path)
            if match:
                groups = match.groupdict()
                return AtsId(ats, groups.get("tenant"), groups["job_id"])
    # Greenhouse jobs embedded on a company's own careers page carry ?gh_jid=<id>.
    gh_jid = parse_qs(parts.query).get("gh_jid")
    if gh_jid and gh_jid[0].isdigit():
        return AtsId("greenhouse", None, gh_jid[0])
    return None
