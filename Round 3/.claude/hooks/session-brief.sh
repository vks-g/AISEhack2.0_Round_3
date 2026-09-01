#!/usr/bin/env bash
# SessionStart: the two-day sprint means no session should have to rediscover
# where things stand. Prints status, the scored runs, and today's slot usage.
cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || exit 0
{
  echo "=== AISEHack Round 3 — session brief ==="
  echo "deadline 3 Sep 2026 | 3 submissions/day, 2 final picks | public LB to beat: 0.917"
  echo
  if [ -f experiments/LOG.md ]; then
    echo "--- scored runs (experiments/LOG.md) ---"
    grep -E '^\| 20' experiments/LOG.md | tail -8
    n=$(grep -cE '^\| 20' experiments/LOG.md)
    echo "($n scored runs on record)"
  fi
  echo
  echo "--- submissions ledger ---"
  grep -A100 'Submission ledger' experiments/LOG.md 2>/dev/null | grep -E '^\| 20' | tail -5
  echo
  if [ ! -d .venv ]; then
    echo "!! .venv is MISSING — rebuild before running anything:"
    echo "   uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements-dev.txt"
  fi
} 2>/dev/null
exit 0
