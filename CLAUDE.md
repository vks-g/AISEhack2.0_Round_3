# AISEHack 2.0 — Polymer Property Prediction

Kaggle-style competition: predict 7 polymer properties from PSMILES strings.
Three rounds; **Round 3 is the active one and has its own CLAUDE.md** with the
operational detail. This file is repo-wide orientation only — when the two
disagree, `Round 3/CLAUDE.md` wins, because its numbers were measured against the
Round 3 data.

## Layout

```
CLAUDE.md              this file
Round 3/               ACTIVE. Read Round 3/CLAUDE.md before doing anything there.
  CLAUDE.md            the operational contract: ground truth, rules, workflow
  .claude/             skills, agents, hooks, settings for the sprint
  src/                 the harness -- source of truth for all modelling
  data/                competition CSVs. Never Read these; use src.data.
  context/             the Kaggle pages, mirrored as markdown
  experiments/LOG.md   every scored run. Read first, always.
  submissions/         exported notebooks
  .venv/               python3.12 + rdkit/lightgbm/xgboost/catboost/torch
"Round 2 /"            NOTE THE TRAILING SPACE in the directory name.
  Submissions/         notebook4_krishna.ipynb, kpp-v1_vetri.ipynb
archive/               zips and screenshots. Superseded by Round 3/context/*.md.
```

`Round 3/data/base_line_model.ipynb` is the host's official baseline: Ridge on
RDKit descriptors, per property, plain `KFold(5, seed=42)`.

## The seven properties

Chain Bandgap (`egc`), Bulk Bandgap (`egb`), Ionisation Energy (`ei`), Electron
Affinity (`eea`), Dielectric Constant (`eps`), Refractive Index (`nc`), Glass
Transition Temperature (`tg`).

`target_type` values are **lowercase** in the CSVs. The metric is the unweighted
mean R² across all seven, so the ~220-row properties count as much as the
4143-row `tg`.

## Running anything

There is no system pandas, rdkit or lightgbm on this machine.

```bash
cd "Round 3" && ./.venv/bin/python -m src.cv --config lgbm
```

A PreToolUse hook blocks bare `python`. Rebuild the venv if missing:

```bash
cd "Round 3" && uv venv --python 3.12 .venv \
  && uv pip install --python .venv/bin/python -r requirements-dev.txt
```

## Rules that apply to every round, and hardened in Round 3

Round 3 §6.2 makes these absolute — violating one is disqualification regardless
of leaderboard position:

- **No external data. No pretrained weights, checkpoints, embeddings, or any
  uploaded artifact.** Everything trains from scratch inside a single Kaggle
  notebook run.
- **Nothing may be read that the notebook run did not itself write.** That
  includes any feature cache, any `.pkl`/`.pt`, and anything under `archive/` or
  `Round 2 /` — those are external data now.
- Seeds set and printed; the pinned notebook version must reproduce the submitted
  score exactly, so no branching on wall-clock time.
- Public GitHub *code* is allowed. Public *weights* are not.
- Output file named exactly `submission.csv`.

Round 2 notebooks predate these clauses. `notebook4_krishna.ipynb` reads
`archive/train.csv` and merges labels from it — under Round 3 rules that is
external data and an instant disqualification. Mine those notebooks for
technique, never copy their I/O.

## Never read into context

`Round 3/data/*.csv` (`smile_r3.csv` is 330 MB / 5.97M rows, `PI1M.csv` 48 MB /
996k), `Round 3/.cache/*.npy`, `*.zip`, and the PNGs in `archive/references/` —
those are screenshots of pages already mirrored as markdown in
`Round 3/context/`. Inspect any CSV with a `./.venv/bin/python -c "import pandas
as pd; ..."` one-liner, or `src.data.load_train()` / `load_test()`.
