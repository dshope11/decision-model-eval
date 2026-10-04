-- Listings store, schema version 1.
--
-- Capture is complete from the first run; anything derivable is a view or waits
-- for a later migration. Timestamps are ISO 8601 UTC text (YYYY-MM-DDTHH:MM:SSZ).
-- JSON columns are TEXT checked with json_valid.

-- Search configuration: the sources, queries and filters that decide which
-- postings can be seen at all.
CREATE TABLE preselection_versions (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    effective_from TEXT NOT NULL,
    config TEXT NOT NULL CHECK (json_valid(config)),
    note TEXT
) STRICT;

-- One fetcher invocation. A failed or partial run must never read as "no change":
-- absence of a listing only means something when the run is ok and complete.
CREATE TABLE runs (
    id INTEGER PRIMARY KEY,
    run_uuid TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('live', 'backfill', 'manual')),
    fetcher TEXT,
    fetcher_version TEXT,
    preselection_version_id INTEGER REFERENCES preselection_versions (id),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('ok', 'partial', 'failed')),
    error TEXT,
    complete INTEGER NOT NULL CHECK (complete IN (0, 1)),
    ingested_at TEXT NOT NULL
) STRICT;

-- One query within a run, with how many results the source reported versus how
-- many were fetched (a capped fetch is a truncated view of the source).
CREATE TABLE run_queries (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs (id),
    query TEXT NOT NULL,
    params TEXT CHECK (params IS NULL OR json_valid(params)),
    results_reported INTEGER,
    results_fetched INTEGER,
    UNIQUE (run_id, query)
) STRICT;

CREATE TABLE companies (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    name_key TEXT NOT NULL UNIQUE  -- casefolded, whitespace-collapsed
) STRICT;

-- A posting's identity within one source. Never merged across sources: the same
-- job seen through an alert and through its ATS is two listings, grouped later.
CREATE TABLE listings (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    source_key TEXT NOT NULL,
    company_id INTEGER REFERENCES companies (id),
    url TEXT,
    ats TEXT,
    ats_tenant TEXT,
    ats_job_id TEXT,
    UNIQUE (source, source_key)
) STRICT;

-- One distinct version of a listing's normalized content. The content hash covers
-- normalized fields only (versioned by hash_version), so volatile fields in the
-- raw record do not mint a new snapshot on every fetch.
CREATE TABLE snapshots (
    id INTEGER PRIMARY KEY,
    listing_id INTEGER NOT NULL REFERENCES listings (id),
    hash_version INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    title TEXT,
    company_name_raw TEXT,
    listing_line TEXT,
    location TEXT,
    comp_min REAL,
    comp_max REAL,
    comp_currency TEXT,
    comp_period TEXT,
    posted_at TEXT,
    posted_at_field TEXT,  -- which source field the date came from; semantics differ by ATS
    level TEXT,
    jd_text TEXT,
    first_seen_at TEXT NOT NULL,
    UNIQUE (listing_id, hash_version, content_hash)
) STRICT;

-- The verbatim source record, stored once per distinct content. Every field not
-- extracted today stays recoverable.
CREATE TABLE payloads (
    hash TEXT PRIMARY KEY,
    raw TEXT NOT NULL CHECK (json_valid(raw))
) STRICT;

-- One observation of a listing by one query. artifact_ref identifies an observed
-- artifact (an alert email's Message-ID) and is NULL for live fetches.
CREATE TABLE sightings (
    id INTEGER PRIMARY KEY,
    run_query_id INTEGER NOT NULL REFERENCES run_queries (id),
    listing_id INTEGER NOT NULL REFERENCES listings (id),
    snapshot_id INTEGER NOT NULL REFERENCES snapshots (id),
    payload_hash TEXT REFERENCES payloads (hash),
    observed_at TEXT NOT NULL,
    artifact_ref TEXT,
    passed_filter INTEGER NOT NULL CHECK (passed_filter IN (0, 1)),
    filter_reason TEXT
) STRICT;

-- A live fetch sees a listing at most once per query per run; an artifact (one
-- email) is one observation however many runs re-read it.
CREATE UNIQUE INDEX sightings_once_per_query
    ON sightings (run_query_id, listing_id) WHERE artifact_ref IS NULL;
CREATE UNIQUE INDEX sightings_once_per_artifact
    ON sightings (listing_id, artifact_ref) WHERE artifact_ref IS NOT NULL;

-- One recorded annotations file. Its UUID makes re-recording the same file a
-- no-op, so a retry cannot silently duplicate decisions, labels or events.
CREATE TABLE record_batches (
    id INTEGER PRIMARY KEY,
    batch_uuid TEXT NOT NULL UNIQUE,
    origin TEXT,  -- e.g. "/discovery 2026-10-03"
    recorded_at TEXT NOT NULL
) STRICT;

-- Every model output: backends under test, the strong-model labeler, and the
-- inline judgments Claude makes during a workflow run (backend 'claude-inline').
CREATE TABLE decisions (
    id INTEGER PRIMARY KEY,
    batch_id INTEGER REFERENCES record_batches (id),
    snapshot_id INTEGER NOT NULL REFERENCES snapshots (id),
    decision_point TEXT NOT NULL,
    backend TEXT NOT NULL,
    model TEXT,
    provider TEXT,
    prompt_template_hash TEXT,
    input_payload TEXT CHECK (input_payload IS NULL OR json_valid(input_payload)),
    output TEXT NOT NULL CHECK (json_valid(output)),
    confidence REAL,
    latency_ms REAL,
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost_usd REAL,
    raw_response TEXT,
    repeat_index INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
) STRICT;

-- Human judgments. sample names the random draw a label belongs to (NULL for an
-- ad hoc label); saw_model_output records whether any model output was visible.
CREATE TABLE labels (
    id INTEGER PRIMARY KEY,
    batch_id INTEGER REFERENCES record_batches (id),
    snapshot_id INTEGER NOT NULL REFERENCES snapshots (id),
    decision_point TEXT NOT NULL,
    labeler TEXT NOT NULL,
    value TEXT NOT NULL CHECK (json_valid(value)),
    sample TEXT,
    saw_model_output INTEGER NOT NULL CHECK (saw_model_output IN (0, 1)),
    rubric_version TEXT,
    labeled_at TEXT NOT NULL,
    note TEXT
) STRICT;

-- Append-only pipeline transitions. The stage a listing reached is derived.
CREATE TABLE pipeline_events (
    id INTEGER PRIMARY KEY,
    batch_id INTEGER REFERENCES record_batches (id),
    listing_id INTEGER NOT NULL REFERENCES listings (id),
    event TEXT NOT NULL CHECK (event IN (
        'promoted', 'queued', 'watching', 'deferred', 'dropped', 'submitted',
        'screen', 'interview', 'offer', 'rejected', 'withdrawn', 'closed'
    )),
    at TEXT NOT NULL,
    channel TEXT,
    drop_reason TEXT,
    evidence TEXT,
    note TEXT,
    recorded_at TEXT NOT NULL
) STRICT;

-- SQLite does not index foreign keys on its own.
CREATE INDEX runs_preselection ON runs (preselection_version_id);
CREATE INDEX run_queries_run ON run_queries (run_id);
CREATE INDEX listings_company ON listings (company_id);
CREATE INDEX sightings_listing ON sightings (listing_id, observed_at);
CREATE INDEX sightings_snapshot ON sightings (snapshot_id);
CREATE INDEX sightings_payload ON sightings (payload_hash);
CREATE INDEX decisions_snapshot ON decisions (snapshot_id);
CREATE INDEX decisions_batch ON decisions (batch_id);
CREATE INDEX labels_snapshot ON labels (snapshot_id);
CREATE INDEX labels_batch ON labels (batch_id);
CREATE INDEX pipeline_events_listing ON pipeline_events (listing_id, at);
CREATE INDEX pipeline_events_batch ON pipeline_events (batch_id);

-- Derived, never stored, so they cannot drift.
CREATE VIEW listing_presence AS
SELECT
    listing_id,
    MIN(observed_at) AS first_seen,
    MAX(observed_at) AS last_seen,
    COUNT(*) AS n_sightings
FROM sightings
GROUP BY listing_id;

CREATE VIEW latest_snapshots AS
SELECT sn.*
FROM snapshots AS sn
WHERE sn.id = (
    SELECT si.snapshot_id
    FROM sightings AS si
    WHERE si.listing_id = sn.listing_id
    ORDER BY si.observed_at DESC, si.id DESC
    LIMIT 1
);
