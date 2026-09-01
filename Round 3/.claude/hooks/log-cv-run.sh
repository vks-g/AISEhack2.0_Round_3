#!/usr/bin/env bash
# PostToolUse[Bash]: a CV run that is not in experiments/LOG.md did not happen.
# Appends the newest experiments/runs/*.json as a table row, exactly once.
input=$(cat)
cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // ""')
printf '%s' "$cmd" | grep -q 'src\.cv' || exit 0

cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || exit 0
latest=$(ls -t experiments/runs/*.json 2>/dev/null | head -1)
[ -n "$latest" ] || exit 0

marker=".claude/.last-logged-run"
[ -f "$marker" ] && [ "$(cat "$marker")" = "$latest" ] && exit 0

row=$(./.venv/bin/python - "$latest" <<'PY' 2>/dev/null
import json, sys, os, time
d = json.load(open(sys.argv[1]))
p = d["per_target"]
cols = " | ".join(f"{p.get(t, float('nan')):.4f}" for t in
                  ["tg","egc","egb","eps","nc","ei","eea"])
stamp = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(sys.argv[1])))
print(f"| {stamp} | {d['config']} | {d['seed']} | **{d['mean_r2']:.4f}** | {cols} "
      f"| {d['noise_floor']:.4f} | {d['wall_seconds']:.0f}s | |")
PY
)
[ -n "$row" ] || exit 0

# Insert into the "Scored runs" table, not at EOF -- the end of LOG.md is the
# dead-ends section. The marker is written by experiments/LOG.md itself.
anchor='<!-- new runs are inserted above this line by .claude/hooks/log-cv-run.sh -->'
if grep -qF "$anchor" experiments/LOG.md; then
  tmp=$(mktemp)
  awk -v row="$row" -v anchor="$anchor" \
    'index($0, anchor) && !done {print row; done=1} {print}' \
    experiments/LOG.md > "$tmp" && mv "$tmp" experiments/LOG.md
else
  printf '%s\n' "$row" >> experiments/LOG.md
fi
printf '%s' "$latest" > "$marker"
jq -n --arg r "$row" '{systemMessage: ("logged to experiments/LOG.md: " + $r)}'
exit 0
