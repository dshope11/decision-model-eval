#!/usr/bin/env bash
# PostToolUse hook: after every agent Edit/Write, run ruff --fix, the pytest
# suite and mypy.
#
# Silent when all three pass. On any failure it prints the failing output to
# stderr and exits 2, which is how a PostToolUse hook gets text back to the
# agent in the same turn. Plain stdout on exit 0 only reaches Claude Code's
# debug log, so a hook that just prints its results is invisible to the agent.
#
# PY points at the dedicated conda env directly, avoiding a shell-init
# dependency; override it with DECISION_MODEL_EVAL_PYTHON.

PY="${DECISION_MODEL_EVAL_PYTHON:-/opt/homebrew/Caskroom/miniforge/base/envs/decision-model-eval/bin/python}"
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

report=""
run() {  # run <label> <command...>
  local label="$1" out
  shift
  if ! out="$("$@" 2>&1)"; then
    report+="== $label failed"$'\n'"$(printf '%s\n' "$out" | tail -30)"$'\n'
  fi
}

run ruff   "$PY" -m ruff check src/ tests/ --fix
run pytest "$PY" -m pytest tests/ -x -q --tb=short
run mypy   "$PY" -m mypy src/ tests/ --no-error-summary

[ -z "$report" ] && exit 0
printf '%s' "$report" >&2
exit 2
