# Experiment log

Append-only. One row per scored run. **Read this first in every session** — a
SessionStart hook prints it for you.

Runs are appended automatically by the `log-cv-run` PostToolUse hook whenever
`src.cv` completes. Runs that are not here did not happen.

The harness prints a **noise floor** (the standard error of the mean score). A
delta smaller than 2x it is not an improvement — say so rather than claiming a win.

## Scored runs

| date | config | seed | mean R² | tg | egc | egb | eps | nc | ei | eea | noise | wall | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-01 | ridge_baseline | 42 | 0.5354 | 0.8187 | 0.2366 | -0.0542 | 0.6055 | 0.7623 | 0.5994 | 0.7796 | 0.0287 | 4s | host baseline equivalent; smoke test only |
| 2026-09-01 | lgbm | 42 | **0.8519** | 0.9095 | 0.9032 | 0.8956 | 0.7248 | 0.8373 | 0.8057 | 0.8875 | 0.0089 | 377s | per-property LightGBM, 2978 features, size-adaptive. **Local reference.** |
| 2026-09-01 | lgbm_physics | 42 | **0.8807** | 0.9091 | 0.9064 | 0.9179 | 0.8109 | 0.8686 | 0.8408 | 0.9114 | 0.0072 | 1178s | + partner features + AFFINE physics blend. **+0.0288 over `lgbm`, 4x the noise floor.** New best. |
<!-- new runs are inserted above this line by .claude/hooks/log-cv-run.sh -->

## Submission ledger — 3/day, 2 final picks, deadline 3 Sep 2026

| date | config / notebook | local CV | public LB | pinned? | shared with hosts? |
|---|---|---|---|---|---|
| earlier | `submissions/aisehack3-1.ipynb` | OOF ~0.903 | **0.883** | ? | ? |
| — | `submissions/round-3-aisehack.ipynb` ("v2") | **never scored** | — | — | — |

Confirm the pinned-version and host-sharing columns for the 0.883 submission
before the deadline. Rules §7.1/§7.2: a mismatched pinned version voids it
regardless of score.

## Ranked plan — where the 0.883 → 0.917 gap is

`tg` (0.910), `egc` (0.903) and `egb` (0.896) are near saturation. The whole gap
is in the four ~220-row properties, which are 4/7 of the metric. Ranked by
expected mean-R² gain per hour:

1. **Affine-fit the physics relations.** *(measured)* `eps` raw 0.336 → fitted
   0.855 at 62% test coverage; `nc` 0.171 → 0.837 at 62%; `egb` 0.892 → 0.928 at
   55%. Both shipped notebooks apply the raw identity. `eps` and `nc` are two of
   the three weakest targets. `src/configs/lgbm_physics.py` implements this.
2. **Tune blend weights on an inner split, never the outer OOF.** The prime
   suspect for the 0.903 OOF → 0.883 LB gap. Same for stacking coefficients on
   220-row properties.
3. **Per-target model selection** rather than one global stack. The seven targets
   have different sample sizes by a factor of 19; one meta-learner choice for all
   of them is a coin flip on the small ones.
4. **SMILES augmentation as regularisation** for `eps`/`nc`/`ei`/`eea` only, plus
   test-time augmentation. Cheap, and it doubles as invariance evidence.
5. **Aux corpus, only if 1–4 are done.** Use `data/PI1M.csv` (996k polymer SMILES
   *with* wildcards), not `smile_r3.csv` (5.97M small molecules, no wildcards,
   ~27 core-hours to featurise). Subsample 200–500k. Budget it before starting.

Not worth doing with two days left: repeat-unit canonicalisation (the data has no
oligomer duplicates — see the invariance section of CLAUDE.md), and anything that
pushes the notebook past a single Kaggle session.

## Decisions / dead ends

Record what did NOT work here so no session retries it.

- **GroupKFold on canonical SMILES is not required** for per-property models.
  *(measured)* Within each `target_type` there are essentially zero duplicate
  canonical polymers, so plain KFold is already leakage-safe. An earlier CLAUDE.md
  mandated GroupKFold on the strength of an unmeasured "0.05–0.08 CV inflation"
  claim. Group only for multi-task models.
- **The raw `eps = nc²` identity is bad** — R² 0.336, not the 0.843 the old docs
  claimed. 0.843-ish is the *affine-fitted* number. Never apply it unfitted.
- **Translation/oligomer canonicalisation buys nothing on this data.** A
  macrocycle key merges only 7 groups out of ~9,000, none differing in heavy-atom
  count. Keep it as an audit tool, not a merge key — it produces false merges.
- **`submissions/round-3-aisehack.ipynb` (v2) is not a safe default.** Unscored,
  drops ingest canonicalisation, fits one physics weight across two populations,
  and very likely exceeds the Kaggle session limit. Score it before trusting it.
