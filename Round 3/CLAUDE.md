# AISEHack 2.0 — Round 3 (Polymer Property Prediction)

Predict 7 polymer properties from PSMILES. **Deadline 3 Sep 2026.**
3 submissions/day, 2 final picks.

Every number in this file marked *(measured)* was produced by running code in
this repo against the real data. Numbers on the Kaggle pages that contradict them
are wrong — one of them (the test row count) has already cost this repo a bug.

## Current status

| | |
|---|---|
| public LB | **0.883** — from `submissions/aisehack3-1.ipynb` (its OOF was ~0.903) |
| target | **0.917** (rank 1, "Sandman"); rank 2 is 0.914 |
| local best | **0.8807** — `src/configs/lgbm_physics.py`, 1178 s *(measured)* |
| local reference | 0.8519 — `src/configs/lgbm.py`, 377 s *(measured)* |
| `submissions/round-3-aisehack.ipynb` | a later "v2". **Never scored.** Do not treat it as the incumbent — see the risks at the bottom of this file. |

Read `experiments/LOG.md` before doing anything. A SessionStart hook prints it.

## Metric

`Score = mean(R² over the 7 target_types)`, **unweighted**.

`tg` has 4143 training rows and `eea` has 221; they count exactly the same. The
four small properties (`eps`, `nc`, `ei`, `eea`, ~220 rows each) are **4/7 of the
score on under 4% of the rows**. Never optimise one global MSE on raw targets —
`tg` ranges to 495 and `nc` to 2.76, so `tg` would dominate and you would silently
lose the properties that matter most. Model per target, or standardise per target
before any shared loss.

## Ground truth *(measured — trust over any doc)*

```
train.csv   7409 rows   (smiles, target, target_type)     long format
test.csv    4940 rows   (id, smiles, target_type)         <- NOT 4497
sample_submission.csv   10 rows -- a format illustration, not a template
```

**`test.csv` has 4940 rows.** 4497 is its count of unique *raw* SMILES; the Kaggle
data page's "4,497 data points" means molecules. A 4497-row submission is
rejected. `submissions/round-3-aisehack.ipynb` still warns `len(test_df) != 4497`
on every correct run — that check is the bug.

**`target_type` values are lowercase**: `tg egc egb eps nc ei eea`. Title-case
matches zero rows and silently scores `nan`.

| target | train n | test n | range | lgbm R² | +physics |
|---|---|---|---|---|---|
| tg | 4143 | 2763 | -109.82 – 495.0 | 0.910 | 0.909 |
| egc | 2028 | 1352 | 0.02 – 9.86 | 0.903 | 0.906 |
| egb | 337 | 224 | 0.51 – 10.11 | 0.896 | **0.918** |
| eps | 229 | 153 | 2.61 – 9.09 | **0.725** | **0.811** |
| nc | 229 | 153 | **1.56 – 2.76** | 0.837 | **0.869** |
| ei | 222 | 148 | 4.03 – 9.84 | **0.806** | **0.841** |
| eea | 221 | 147 | 0.39 – 5.14 | 0.888 | **0.911** |

`nc` is 1.56–2.76, not the "1.5–1.7" this repo used to claim.

Every PSMILES has exactly two `*` wildcards. All 10,605 unique SMILES parse in
RDKit. Canonicalisation: train 6565 raw → 5920; test 4497 raw → 4133. Three
`(smiles, target_type)` keys in train carry conflicting targets (all `tg`);
`src/data.py` averages them, giving 7405 rows.

Only **2 of 4940** test rows have their `(canonical smiles, target_type)` present
in train — there are no free lookups. 1063/4133 (25.7%) of test polymers appear
in train under *some* property, which is what makes partner features work.

### Splits — per-property KFold, not GroupKFold

Within a single `target_type` there are essentially no duplicate canonical
polymers *(measured)*: tg 4143 rows / 4139 canonical, and egc, egb, eps, nc, ei,
eea have **zero** duplicates each. So plain `KFold` over one property's rows is
already leakage-safe, and it is what the host's baseline does. An earlier version
of this file mandated GroupKFold and claimed KFold inflates CV by 0.05–0.08 —
that was never measured and is false for per-property models.

Group only for a **multi-task** model trained on the full long table, where one
polymer contributes up to six rows (6150 polymers have 1 property, 157 have 2,
126 have 3, 100 have 4, 28 have 5, 4 have 6). `src.splits.grouped_folds()`.

## Hard rules — violating these is disqualification, not a bad score

From `context/rules.md` §6.2 and §7. These are absolute:

- **No external data.** Only the official competition files. `archive/`,
  `Round 2 /`, and anything produced outside the notebook run are external data.
- **No pretrained weights, checkpoints, embeddings or cached artifacts.** This
  kills polyBERT, ChemBERTa, MolFormer, any HuggingFace checkpoint, and any
  `.pt`/`.pkl`/`.npy` you produce locally and upload.
- **One notebook execution does everything**: load → preprocess → split → train →
  infer → write `submission.csv`. No manual steps, no uploaded artifacts.
- Public GitHub **code** may be imported and executed. Public **weights** may not.
- **Seeds set and printed.** The pinned/default notebook version must reproduce
  the submitted score *exactly*, so nothing may branch on wall-clock time. A
  2-hour training budget is a reproducibility violation, not a nice-to-have.
- Notebook shared with view access to all five hosts (Rohit Batra IITM,
  Rahulsundar, LaksmanN, VIJITH P, shreyasri0301), linked in the submission
  description, and **pinned to the version that produced the score**.
- Output file named exactly `submission.csv`.

The local `.cache/` feature store is fine — it is local only, and the export
strips it. `src/features.py` recomputes from scratch in ~25 s, so the notebook
never needs a cache.

## How to run anything

**There is no system pandas, rdkit or lightgbm.** Use the project venv:

```bash
cd "Round 3"
./.venv/bin/python -m src.cv --config lgbm             # score a config (the gate)
./.venv/bin/python -m src.predict --config lgbm        # full fit -> submission.csv
./.venv/bin/python -m src.check_submission sub.csv     # validate before uploading
./.venv/bin/python -m src.invariance --config lgbm     # invariance certificate
./.venv/bin/python -m src.explain --config lgbm        # per-target SHAP report
```

A hook blocks bare `python`. If `.venv` is missing:
`uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements-dev.txt`

`src/` is the source of truth; `submissions/*.ipynb` are exports of it. A config
is two functions — `fit(train_df, seed, targets=None)` and `predict(state, df)` —
and CV, inference, invariance and explainability all derive from those, so the
model has one implementation and what CV measured is what gets submitted. See
`src/configs/lgbm.py`.

*(measured)* `src.predict --config lgbm` full-fits in 55 s and writes a
submission that passes `src.check_submission`.

## Workflow

1. Every idea is scored by `./.venv/bin/python -m src.cv --config <name>` before
   it gets a submission slot. Never submit an unscored change.
2. The harness prints a **noise floor**. A delta under 2x it is not an
   improvement. `eps` is the noisiest target (fold std ~0.14) — confirm any win
   there with `--seed 7`.
3. Runs are auto-logged to `experiments/LOG.md` by a PostToolUse hook. Record
   dead ends there too, so no session repeats them.
4. Gate every upload through `src.check_submission` and the `submission-check`
   skill.

## The two judged themes

**Invariance.** *(measured)* The dataset contains **no genuine oligomer
duplicates** and only 7 borderline translational groups out of ~9,000 polymers.
So this is a rubric deliverable, not leaderboard points — do not burn a day on
repeat-unit reduction. What the pipeline guarantees: permutational invariance is
**exact** (every feature derives from the canonical SMILES), translational and
repetition invariance are measured and reported by `src/invariance.py`. See the
`invariance-audit` skill.

**Explainability.** Per-target TreeSHAP via LightGBM's own `pred_contrib=True` —
no `shap` dependency to fail inside the notebook. Attribute by feature family,
not by individual Morgan bit. See the `explainability` skill. Cheap to produce
and it is half the rubric.

## Where the remaining headroom is

`tg`, `egc` and `egb` are all ~0.90 and close to saturated. The gap is the four
small properties, and physics reaches three of them. **Always fit the relations
affine — never apply the raw identity** *(measured)*:

| target | expression | raw R² | fitted R² | test coverage |
|---|---|---|---|---|
| ei | `egc + eea` | 0.9629 | 0.9650 | 37% |
| eea | `ei - egc` | 0.9710 | 0.9727 | 35% |
| egb | `egc` | 0.8922 | **0.9282** | 55% |
| eps | `nc²` | **0.3364** | **0.8553** | 62% |
| nc | `sqrt(eps)` | **0.1708** | **0.8370** | 62% |

Both shipped notebooks apply the raw form. **`src/configs/lgbm_physics.py` does
it correctly and is worth +0.0288 mean R² over plain `lgbm` — 4x the noise
floor** *(measured)*, concentrated exactly where predicted: eps +0.086,
ei +0.035, nc +0.031, eea +0.024, egb +0.022, tg flat. See the `physics-blend`
skill for how to fit and tune the weight without leaking.

That is the single biggest validated lever in this repo. The next gains come
from stacking several base models on top of it — which is what the 0.883
notebook does, without the fitted physics.

The OOF 0.903 → LB 0.883 gap is most likely OOF-tuned blend weights and stacking
on 220-row properties. Tune blend weights on an **inner** split of the training
fold, never on the outer OOF.

## Compute discipline

Kaggle has a hard runtime limit and an over-budget notebook produces **no
submission at all**. *(measured on this machine, 11 cores)*: featurising all
12,345 molecules takes ~25 s; a full 10/15-fold CV of `lgbm` is 377 s.

RDKit-parsing all 5.97M aux SMILES would take ~27 core-hours — it will not
finish. If you use auxiliary data at all, prefer **`data/PI1M.csv`** (995,800
rows, column `SMILES`, real polymer SMILES *with* `*` wildcards, ~4.5 core-hours
for the full set) over `data/smile_r3.csv` (5,973,370 rows, column `smiles`,
small molecules with **no** wildcards). PI1M matches the task distribution and is
6x cheaper. Both are official competition files. Subsample to 200–500k, use it for
something cheap, and time it locally first.

## Files you must NOT read into context

| Path | Why |
|---|---|
| `data/smile_r3.csv` | 330 MB, 5,973,370 rows |
| `data/PI1M.csv` | 48 MB, 995,800 rows |
| `data/train.csv`, `data/test.csv` | use `src.data.load_train()` / `load_test()` |
| `.cache/*.npy` | 107 MB feature matrix |
| `archive/`, `*.zip` | duplicates of `context/*.md`, and external data |

A hook blocks these and tells you the pandas one-liner instead. Everything about
the competition is in `context/*.md` — read those, not the PNGs.

## Known risks in `submissions/round-3-aisehack.ipynb` (the unscored v2)

Do not submit it without scoring it first. Relative to the 0.883 incumbent it:
drops ingest canonicalisation, weakening the invariance claim from exact to
approximate; replaces the two-pass physics blend with one coalesced pass whose
single weight is fitted across two populations with very different physics R²,
then raises shrinkage to 0.90 for `ei`/`eea` so it touches 100% of those rows;
adds a hard `import shap` that v1 deliberately engineered away; truncates
molecules at `MAX_ATOMS=60` in the MPNN, affecting 6.2% of polymers; carries a
2-hour SSL time budget that breaks §7.2 reproducibility if enabled; and very
likely exceeds a single Kaggle session.
