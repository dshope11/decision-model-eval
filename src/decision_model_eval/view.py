"""The listing view: what a listing-line judgment sees, for human and model alike.

A label is only comparable to a model decision when both were made on the same
information, so the blind sheet (``dme sample``) and the triage list
(``dme surfaced``) render every listing through this one function.

View v2 (rubric ``d1-v2``): company, title and a cleaned location, plus a
details list - work mode, level, pay and role category - drawn from the stored
snapshot and the verbatim source record. A field the source does not carry is
shown as ``?`` (work mode, level) or left out (pay, role) rather than guessed.

View v1 (rubric ``d1-v1``) was ``company :: title :: location`` with the location
as the source joined it.

Known limit: work mode and role category come from the source record, which is not
part of the snapshot content hash. A source that changed only those fields would
render the same snapshot differently before and after; none had on 2026-10-04.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

VIEW_VERSION = "v2"

# WaaS gives countries as ISO 3166 codes; these are the ones its listings use.
# US stays "US". Codes not listed here are shown as given.
_COUNTRIES = {
    "AE": "UAE", "AR": "Argentina", "AT": "Austria", "AU": "Australia",
    "AZ": "Azerbaijan", "BE": "Belgium", "BG": "Bulgaria", "BR": "Brazil",
    "CA": "Canada", "CH": "Switzerland", "CL": "Chile", "CN": "China",
    "CO": "Colombia", "CZ": "Czechia", "DE": "Germany", "DK": "Denmark",
    "ES": "Spain", "FR": "France", "GB": "UK", "GI": "Gibraltar", "GR": "Greece",
    "HK": "Hong Kong", "HU": "Hungary", "ID": "Indonesia", "IE": "Ireland",
    "IL": "Israel", "IN": "India", "JM": "Jamaica", "JO": "Jordan", "JP": "Japan",
    "KR": "South Korea", "KZ": "Kazakhstan", "LT": "Lithuania", "MA": "Morocco",
    "MX": "Mexico", "MY": "Malaysia", "NA": "Namibia", "NL": "Netherlands",
    "PA": "Panama", "PH": "Philippines", "PK": "Pakistan", "PL": "Poland",
    "PT": "Portugal", "RO": "Romania", "RS": "Serbia", "SA": "Saudi Arabia",
    "SE": "Sweden", "SG": "Singapore", "SY": "Syria", "TH": "Thailand",
    "TR": "Turkey", "TW": "Taiwan", "UA": "Ukraine", "UG": "Uganda",
    "VN": "Vietnam", "ZA": "South Africa", "ZM": "Zambia",
}
_WAAS_WORK = {"no": "on-site", "yes": "remote ok", "only": "remote only"}
_GETRO_WORK = {"on_site": "on-site", "remote": "remote", "hybrid": "hybrid"}
_ROLE_TYPES = {
    "fs": "full-stack", "be": "backend", "fe": "frontend", "ml": "ML",
    "data_sci": "data science", "eng_mgmt": "eng management", "ui_ux": "UI/UX",
    "devops": "DevOps", "qa": "QA", "hw": "hardware", "ios": "iOS",
}
_PERIODS = {"year": "/yr", "month": "/mo", "hour": "/hr"}


@dataclass(frozen=True)
class ListingView:
    company: str
    title: str
    location: str
    details: tuple[str, ...]

    @property
    def line(self) -> str:
        return f"{self.company} :: {self.title} :: {self.location}"


def listing_view(conn: sqlite3.Connection, snapshot_id: int) -> ListingView:
    """The v2 view of one snapshot, using the latest source record seen with it."""
    row = conn.execute(
        """
        SELECT l.source, sn.company_name_raw, sn.title, sn.location, sn.level,
               sn.comp_min, sn.comp_max, sn.comp_currency, sn.comp_period
        FROM snapshots sn JOIN listings l ON l.id = sn.listing_id
        WHERE sn.id = ?
        """,
        (snapshot_id,),
    ).fetchone()
    payload = conn.execute(
        """
        SELECT p.raw FROM sightings si JOIN payloads p ON p.hash = si.payload_hash
        WHERE si.snapshot_id = ? ORDER BY si.observed_at DESC, si.id DESC LIMIT 1
        """,
        (snapshot_id,),
    ).fetchone()
    raw: dict[str, Any] = json.loads(payload["raw"]) if payload else {}
    source = row["source"]
    if source == "waas":
        job = raw.get("job") or {}
        location = _clean_locations(_split(row["location"]), expand_countries=True)
        work = _WAAS_WORK.get(str(job.get("remote")))
        level = _waas_level(job)
        role = _waas_role(job)
    elif source.startswith("getro:"):
        given = [loc for loc in raw.get("locations") or [] if isinstance(loc, str)]
        location = _clean_locations(given or _split(row["location"]))
        work = _GETRO_WORK.get(str(raw.get("work_mode")))
        level = row["level"].replace("_", "-") if row["level"] else None
        role = None
    else:
        location = _clean_locations(_split(row["location"]))
        work = level = role = None
    details = [f"work: {work or '?'}", f"level: {level or '?'}"]
    pay = _pay(row["comp_min"], row["comp_max"], row["comp_currency"], row["comp_period"])
    if pay:
        details.append(f"pay: {pay}")
    if role:
        details.append(f"role: {role}")
    return ListingView(row["company_name_raw"] or "?", row["title"] or "?",
                       location or "?", tuple(details))


def _split(location: str | None) -> list[str]:
    return [part.strip() for part in (location or "").split(";")]


def _clean_locations(entries: list[str], expand_countries: bool = False) -> str:
    """Distinct entries, dropping any whose parts all appear in an entry with more
    parts ("US" and "NY, US" beside "New York, NY, US" leave the city; "Jakarta, ID"
    beside "Jakarta, Jakarta, ID" leaves the latter). Matching parts rather than
    suffixes also clears malformed lists that repeat each part as its own entry.
    The cost: a bare country code that collides with a state ("CA" for Canada beside
    "San Francisco, CA, US") is dropped too - 3 of 3,747 WaaS listings on 2026-10-04,
    against showing a loose California "CA" as Canada."""
    distinct = list(dict.fromkeys(e for e in entries if e))
    parts = {e: e.split(", ") for e in distinct}
    kept = [e for e in distinct
            if not any(len(parts[o]) > len(parts[e]) and set(parts[e]) <= set(parts[o])
                       for o in distinct)]
    if expand_countries:
        kept = [_expand_country(e) for e in kept]
    return "; ".join(kept)


def _expand_country(entry: str) -> str:
    head, sep, code = entry.rpartition(", ")
    return f"{head}{sep}{_COUNTRIES.get(code, code)}"


def _waas_level(job: dict[str, Any]) -> str | None:
    parts = []
    if job.get("job_type") and job["job_type"] != "fulltime":
        parts.append(str(job["job_type"]))
    if isinstance(job.get("min_experience"), int):
        parts.append(f"{job['min_experience']}+ yrs")
    return ", ".join(parts) or None


def _waas_role(job: dict[str, Any]) -> str | None:
    role = job.get("role")
    if not role:
        return None
    # Only the subtype list that belongs to the role: an ops role can carry a stale eng_type.
    types = [t for t in job.get(f"{role}_type") or [] if isinstance(t, str)]
    named = [_ROLE_TYPES.get(t, t.replace("_", " ")) for t in types]
    return f"{role}: {', '.join(named)}" if named else str(role)


def _pay(low: float | None, high: float | None, currency: str | None,
         period: str | None) -> str | None:
    if low is None and high is None:
        return None
    prefix = "$" if currency in (None, "USD") else f"{currency} "
    suffix = _PERIODS.get(period or "", f"/{period}" if period else "")

    def amount(value: float) -> str:
        return f"{value / 1000:.4g}k" if value >= 1000 else f"{value:.4g}"

    if low is not None and high is not None and high != low:
        text = f"{prefix}{amount(low)}-{amount(high)}"
    elif low is not None:
        text = f"{prefix}{amount(low)}" + ("" if high is not None else "+")
    else:
        assert high is not None
        text = f"up to {prefix}{amount(high)}"
    return text + suffix
