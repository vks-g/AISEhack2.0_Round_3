#!/usr/bin/env bash
# PreToolUse[Bash]: refuse bare `python` / `python3`.
# There is no system pandas/rdkit/lightgbm on this machine. Everything must run
# through ./.venv/bin/python or it fails with ModuleNotFoundError.
input=$(cat)
cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // ""')

# Match a bare python/python3 at the start of the command or after a shell operator,
# but not .venv/bin/python, /usr/bin/python, uv run python, etc.
if printf '%s' "$cmd" | grep -Eq '(^|[;&|]|&&|\|\||\bthen |\bdo )[[:space:]]*python3?[[:space:]]'; then
  if ! printf '%s' "$cmd" | grep -Eq '(\.venv/bin/python|/bin/python|uv run|VIRTUAL_ENV)'; then
    jq -n '{
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: "deny",
        permissionDecisionReason: "Bare `python`/`python3` has no pandas, rdkit, lightgbm, xgboost, catboost or torch on this machine — it will fail with ModuleNotFoundError. Use the project virtualenv instead:\n\n    ./.venv/bin/python -m src.cv --config <name>\n    ./.venv/bin/python -c \"...\"\n\nRun from the Round 3 directory. If the venv is missing, rebuild it with:\n    uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements-dev.txt"
      }
    }'
    exit 0
  fi
fi
exit 0
