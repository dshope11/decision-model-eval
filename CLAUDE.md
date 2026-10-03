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
| Backends under test | Jev 1.13 (`typesafe/jev-1.13`) and Claude Haiku 4.5, both through OpenRouter | One key, one credit pool, uniform telemetry; a purpose-built decision model vs a small general model |
| Labeler | Claude Opus 5.5 through OpenRouter labels every item; a weekly blind human sample of ~30 measures its agreement | Full-volume labels from the strong model, corrected by a scale factor from the human control sample |
| Version pinning | Every model id pinned; no `-latest` aliases or routers | A measurement against a moving target is not reproducible. Enforced in `tests/test_backends.py` |
| Provider pinning | OpenRouter provider pinned per backend and logged per call | Unpinned routing spreads one model id across providers with different feature support (some Bedrock endpoints for Opus 5.5 lack structured outputs) |
| Probabilities | Haiku and Opus: per-option probabilities via strict JSON-schema structured output. Jev: its own typed-question endpoint | Jev returns calibrated probabilities natively; the general models must state them |
| Listings store | SQLite, outside the repo, never committed | Holds scraped postings and private labels. `config.py` rejects a `db_path` inside the repo |
| Threshold protocol | Select on one split, freeze, report on another; the oracle row is shown only as a labeled biased upper bound | An argmax over many thresholds rides upward fluctuations |
| Baselines | A non-model baseline (keyword rules) wherever one is natural; backends reported as a gain over it | A model result means little without the simple alternative beside it |
| First result | SciFact claim verification as a public control set, before any live-pipeline result | Public, labeled, and very likely seen in training: a friendly control with a narrow claim |
| In code, not in a model | Dedup, baseline diff, counting requirements, dates | Deterministic work is not a judgment |

---

## Directory Structure

```
decision-model-eval/
  config.yaml                 <- run configuration, validated at load
  src/decision_model_eval/
    backends.py               <- BACKENDS registry: pinned model ids, provider, api, role
    config.py                 <- EvalConfig dataclass + YAML loader (rejects unknown keys)
    stats.py                  <- clopper_pearson
    thresholds.py             <- frozen / oracle / baseline-matched threshold protocol
  tests/                      <- pytest; one file per module
  .claude/hooks/check.sh      <- ruff + pytest + mypy after every agent edit
```

---

## Development

```bash
conda activate decision-model-eval
pip install -e .
pytest
ruff check src/ tests/
mypy src/ tests/
```

`.claude/hooks/check.sh` runs all three after every Edit/Write and returns failures to
the agent in the same turn.

---

## Conventions

- **Source code is plain ASCII** - no Unicode symbols in code, comments, or docstrings.
- **US English** in all prose, docs, and comments.
- **Secrets come from the environment** (`OPENROUTER_API_KEY`), never from a file in the repo.
- **Every decision record is self-describing** (backend, pinned model version, provider,
  prompt-template hash), so analysis can refuse to mix versions.
