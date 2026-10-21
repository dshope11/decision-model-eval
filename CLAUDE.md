# decision-model-eval

Do cheap decision models hold decision quality at the judgment points of a real
agentic workflow, and are their probabilities honest enough to set working points
on? Measured per decision point, per backend, on a live job-search pipeline: each
judgment the workflow makes is pulled out into an explicit, logged model call,
scored against labels, and compared with a non-model baseline.

---

## Locked Decisions - Do Not Re-Derive

| Decision | Choice | Reason |
|---|---|---|
| Backends under test | Jev 1.14 (`typesafe/jev-1.14`) and Claude Haiku 5.5 (`anthropic/claude-haiku-5.5`), both through OpenRouter | One key, one credit pool, uniform telemetry; a purpose-built decision model vs a small general model |
| Labeler | Claude Opus 5.5 through OpenRouter is the reference labeler for the evaluated items; a weekly blind human sample of ~30 measures its agreement | Full-volume labels from the strong model, corrected by a scale factor from the human control sample |
| Version pinning | Every model id pinned, and the reasoning effort wherever a model has one (Haiku 5.5: `medium`, its default); no `-latest` aliases or routers. Recorded `model` = `<resolved id>/effort=<level>` | A measurement against a moving target is not reproducible, and effort moves the answers as much as a version does. Enforced in `tests/test_backends.py`; `decide()` refuses a messages backend with no pinned effort and a response that names another model or provider |
| Inline baseline identity | Judgments Claude makes inside a workflow session are backend `claude-inline`, with `model` = `<model id>/effort=<level>` (e.g. `claude-opus-5-5/effort=high`) | Reasoning effort changes the judgments as much as a model version does; without it, runs at different levels silently mix into one baseline. Enforced by `dme record` |
| Provider pinning | OpenRouter provider pinned per backend and logged per call | Unpinned routing spreads one model id across providers with different feature support (some Bedrock endpoints for Opus 5.5 lack structured outputs) |
| Probabilities | Haiku and Opus: per-option probabilities via strict JSON-schema structured output, through OpenRouter's Anthropic-format `/v1/messages` endpoint with `output_config.effort`; recorded as stated, not renormalized. Jev: its own typed-question endpoint | Jev returns per-option probabilities natively; the general models must state them. Whether either is calibrated is the question under test. `/v1/messages` is the route where effort reaches the model (`reasoning.effort` on chat completions had no measurable effect, 2026-10-20) |
| Decision calls | `decide(point, state, question)`: one question per call, Jev's question format for every backend; `temperature` / `top_p` never sent | Each probability comes from a call that saw only its question; prompt caching keeps a shared state cheap. The 5.5 models reject sampling parameters and OpenRouter drops them silently |
| Decision log | One JSONL line per attempt, failures included, fsynced as the call returns (format 2 of `dme record`'s format). Listing decisions go into the store with `dme record`; other items (SciFact) stay as JSONL | A paid call is on disk as soon as it returns, and the log is the replay source. The store's `decisions` rows need a listing snapshot |
| Listings store | SQLite, outside the repo, never committed | Holds scraped postings and private labels. `config.py` rejects a `db_path` inside the repo |
| Threshold protocol | Select on one split, freeze, report on another; the oracle row is shown only as a labeled biased upper bound. Every distinct score is a candidate threshold | An argmax over many thresholds rides upward fluctuations; at a few hundred items, a thinned grid misplaces recall-matched points |
| Working-point objective | An asymmetric cost: a missed good listing costs far more than one wasted full read | The two errors are not symmetric for the person acting on the decision |
| Baselines | A non-model baseline (keyword rules) wherever one is natural; backends reported as a gain over it | A model result means little without the simple alternative beside it |
| First result | SciFact claim verification as a public control set, before any live-pipeline result | Public, labeled, and very likely seen in training: a friendly control with a narrow claim |
| In code, not in a model | Dedup, baseline diff, counting requirements, dates | Deterministic work is not a judgment |

---

## Directory Structure

```
decision-model-eval/
  config.yaml                 <- run configuration, validated at load
  src/decision_model_eval/
    backends.py               <- BACKENDS registry: pinned model ids, effort, provider, api, role
    questions.py              <- typed questions (noul / choice / score) and their two hashes
    client.py                 <- decide(): the Jev and Anthropic-messages adapters, retries, privacy strip
    decision_log.py           <- append-only JSONL writer, one line per attempt
    replay.py                 <- re-ask a decision log's inputs against any backend
    config.py                 <- EvalConfig dataclass + YAML loader (rejects unknown keys)
    stats.py                  <- clopper_pearson
    thresholds.py             <- frozen / oracle / baseline-matched threshold protocol
    migrations/0001_init.sql  <- listings store schema; PRAGMA user_version tracks the version
    store.py                  <- connect (foreign keys on, WAL), migrate, VACUUM INTO backups
    ingest.py                 <- fetch-script JSONL -> runs, listings, snapshots, sightings
    record.py                 <- decisions / labels / pipeline events JSONL, and decision logs
    surfaced.py               <- what recent runs let through, with store ids, new vs seen
    sample.py                 <- seeded blind label draws, the label sheet, sheet -> label records
    view.py                   <- the listing view shown to labelers and triage alike
    ats.py                    <- ATS identity (greenhouse / ashby / lever / waas) from a URL
    jsonl.py                  <- strict JSONL reading and field validation
    cli.py                    <- the `dme` command
  scripts/                    <- interim stand-alone helpers, until dme label reads D2/D5:
    build_d2_sheet.py         <-   the stage-2 (D2 + D5) sheet from a recorded D1 sheet
    jd_to_text.py             <-   a fetched non-WaaS JD (iCIMS / SmartRecruiters) -> plain text
  tests/                      <- pytest; one file per module
  .claude/hooks/check.sh      <- ruff + pytest + mypy after every agent edit
```

---

## Development

```bash
conda create -n decision-model-eval python=3.12
conda activate decision-model-eval
pip install -e . -r requirements-dev.txt
pytest
ruff check src/ tests/
mypy src/ tests/
```

`.claude/hooks/check.sh` runs all three after every Edit/Write and returns failures to
the agent in the same turn. It only fires in a session started in this repo; from
another working directory, run the three by hand.

The store is driven by the `dme` command (`dme --help`): `init`, `ingest`, `surfaced`,
`record`, `replay`, `sample`, `label`, `preselection add|list`, `backup`, `stats`. Model
calls go through `client.decide()`; its log format is in the docstrings of `client.py`
and `decision_log.py`, and tests replay canned live responses (`tests/conftest.py`), so
the suite never touches the network. The JSONL formats
it reads are specified in the docstrings of `ingest.py` and `record.py`; the label-sheet
format in `sample.py`. **Blind samples draw from everything the runs sighted, filter
rejects included, and the sheet shows the listing view only** - never a tier, a filter
verdict or any model output, which is what makes `saw_model_output: false` true. The
view (`view.py`) is shared by the sheets and `dme surfaced`, so blind labels and
inline triage decisions are made on the same fields; its version is part of the
rubric version, and `dme record` requires inline decisions to name it.

---

## Listings store rules

- **Capture is complete; anything derivable waits.** Adding a column is a new migration
  file; data that was never captured cannot be recovered. Every fetched record is kept,
  filter rejects included, with its verbatim source record.
- **Never edit an applied migration.** Add `0002_...sql`. `store.migrations()` refuses gaps.
- **Listing identity is per source and never merged.** Grouping one posting seen through
  several sources is a later, separate layer built from the stored URLs and ATS ids.
- **Ingest is atomic and idempotent** (one transaction per file, keyed by `run_uuid` /
  `batch_uuid`), and bad input fails loudly before anything is written.
- **A failed or partial run is recorded as such**, so a listing's absence only means
  something when its run was `ok` and `complete`.

---

## Conventions

- **Source code is plain ASCII** - no Unicode symbols in code, comments, or docstrings.
- **US English** in all prose, docs, and comments.
- **Secrets come from the environment** (`OPENROUTER_API_KEY`), never from a file in the repo.
- **Every decision record is self-describing** (backend, pinned model version and effort,
  provider, `question_hash` for the wording, `prompt_template_hash` for what was sent), so
  analysis can refuse to mix versions.
