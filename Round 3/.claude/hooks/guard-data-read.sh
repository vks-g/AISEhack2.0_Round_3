#!/usr/bin/env bash
# PreToolUse[Read]: never pull competition CSVs into context.
# smile_r3.csv is 330 MB / 5,973,370 rows and PI1M.csv is 48 MB / 995,800 rows;
# even train.csv at 7409 rows is pure waste as raw text.
input=$(cat)
path=$(printf '%s' "$input" | jq -r '.tool_input.file_path // ""')

case "$path" in
  *smile_r3.csv|*PI1M.csv|*/data/train.csv|*/data/test.csv|*sample_submission.csv|*.npy|*.npz)
    reason=$(printf '%s' \
"Do not Read ${path} — competition data belongs in pandas, not in context (smile_r3.csv alone is 330 MB / 5,973,370 rows).

Inspect it like this instead:
    ./.venv/bin/python -c \"import pandas as pd; d=pd.read_csv(FILE); print(d.shape); print(d.head())\"

Most facts you want are already recorded in Round 3/CLAUDE.md under 'Ground truth', and src/data.py returns both frames canonicalised via load_train() / load_test().")
    jq -n --arg r "$reason" '{
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: "deny",
        permissionDecisionReason: $r
      }
    }'
    exit 0 ;;
esac
exit 0
