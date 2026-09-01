---
name: cv-harness
description: Score a modelling idea on the exact competition metric before it costs a submission slot. Use whenever evaluating, comparing, tuning, or claiming an improvement on the polymer property task.
---

# Scoring a change

Never eyeball a change. Run it.

```bash
cd "Round 3" && ./.venv/bin/python -m src.cv --config <name>
```

`python` alone is not on PATH and system `python3` has no pandas or rdkit. Always
`./.venv/bin/python`. A PreToolUse hook will stop you if you forget.

## The config API — two functions, nothing else

`src/configs/<name>.py`:

```python
def fit(train_df, seed=42, targets=None) -> state
def predict(state, df) -> np.ndarray          # aligned to df's rows
```

`targets` lets CV fit only the property it is about to score; pass `None` for the
full fit that produces a submission. Cross-validation, the invariance audit, the
explainability report and the Kaggle export all derive from these two functions,
so the model has exactly one implementation and CV cannot drift from submission.

`train_df` / `df` carry `smiles`, `target_type`, `canon` (canonical SMILES) and,
for train, `target`. Featurise from `canon`, never from `smiles` — that is what
makes the pipeline invariant to how a polymer was written.

## The split, and why it is not GroupKFold

Per-property `KFold`. **Measured:** within a single `target_type` there are
essentially no duplicate canonical polymers — tg 4143 rows / 4139 canonical,
and egc/egb/eps/nc/ei/eea have zero duplicates each. So a plain KFold over one
property's rows is already leakage-safe, and it matches the host's own baseline.

Group only when a single model trains on the full long table at once (a
multi-task net), where one polymer contributes up to six rows. `src.splits`
provides `grouped_folds()` for exactly that case.

Fold counts adapt: 10 folds normally, 15 when a property has fewer than 400 rows.
The four ~220-row properties are 4/7 of the score and their OOF is noisy; more
folds buys a bigger training fraction per fold for negligible time.

## Reading the result

The harness prints a per-target table, the mean, and a **noise floor** — the
standard error of the mean score, computed from the per-fold spread. Rules:

1. A delta smaller than 2x the printed noise floor is not an improvement.
   Say so plainly rather than reporting it as a win.
2. Report the per-target table, never the mean alone. A mean that rises while
   `eps` or `nc` collapses is a worse model; the metric weights all seven equally.
3. `eps` is the noisiest target (fold std ~0.14). Confirm any win there with a
   second seed before believing it: `--seed 7`.
4. Local CV far above ~0.92 means leakage, not brilliance. The public LB leader
   is 0.917 and the team's own best notebook scored 0.883.
5. Time every run. A config that cannot complete a full fit inside a Kaggle
   session is not a candidate however good its CV.

## Current reference numbers

| config | mean R² | tg | egc | egb | eps | nc | ei | eea | wall |
|---|---|---|---|---|---|---|---|---|---|
| `ridge_baseline` | 0.5354 | 0.819 | 0.237 | -0.054 | 0.606 | 0.762 | 0.599 | 0.780 | 4s |
| `lgbm` | 0.8519 | 0.910 | 0.903 | 0.896 | 0.725 | 0.837 | 0.806 | 0.888 | 377s |
| `lgbm_physics` | **0.8807** | 0.909 | 0.906 | 0.918 | 0.811 | 0.869 | 0.841 | 0.911 | 1178s |

## Turning a scored config into a submission

```bash
./.venv/bin/python -m src.predict --config <name> --out submission.csv
./.venv/bin/python -m src.check_submission submission.csv
```

`src.predict` calls the same `fit`/`predict` pair on the whole training set and
clips each property to its observed range +5%. `lgbm` full-fits in 55 s.
Always run `check_submission` afterwards — see the `submission-check` skill.

`experiments/LOG.md` is the running record and a PostToolUse hook appends every
run to it automatically. Read it before proposing anything — the dead-ends
section exists so no session repeats a failed idea.

## Where the headroom is

`tg` and `egc` are near 0.91 and close to saturated. The gap is in the four small
properties, and `lgbm_physics` already collected +0.0288 mean R² there by fitting
the physics relations affine instead of raw (see the `physics-blend` skill).
Spend remaining effort on those four targets, not on `tg`.
