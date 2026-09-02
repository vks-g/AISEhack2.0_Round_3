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
| 2026-09-01 | lgbm (partners+physics-feat) | 42 | 0.8645 | 0.906 | 0.904 | 0.900 | 0.781 | 0.853 | 0.811 | 0.883 | — | 343s | base model only, before the physics blend |
| 2026-09-01 | xgb  (partners+physics-feat) | 42 | 0.8629 | | | | | | | | — | 232s | base model only |
| 2026-09-01 | cb   (partners+physics-feat) | 42 | 0.8677 | | | | | | | | — | 486s | **best single base model** |
| 2026-09-01 | lgbm+xgb+cb → stack → staged physics | 42 | **0.8974** | 0.9064 | 0.9073 | 0.9327 | 0.8446 | 0.8827 | 0.8788 | 0.9295 | — | 1065s | per-property folds, partner feats, staged+masked physics. **New best.** |
| 2026-09-01 | mtnn (multi-task NN, 1 seed) | 42 | 0.8719 | 0.880 | 0.870 | 0.929 | 0.808 | 0.894 | 0.843 | 0.880 | — | 1098s | **best single model**; complementary to the GBDTs |
| 2026-09-01 | cnn (SMILES 1-D CNN, 1 seed) | 42 | 0.8206 | 0.818 | 0.835 | 0.862 | 0.724 | 0.833 | 0.794 | 0.880 | — | 1276s | dropped — see dead ends |
| 2026-09-01 | lgbm+xgb+cb+mtnn → stack → staged physics | 42 | **0.9040** | 0.909 | 0.910 | 0.941 | 0.850 | 0.903 | 0.884 | 0.931 | — | — | **shipped config** (`submissions/final.ipynb`) |
| 2026-09-01 | lgbm+xgb+cb+mtnn → stack → staged physics → partner regression | 42 | **0.9070** | 0.9088 | 0.9110 | 0.9414 | 0.8549 | 0.9119 | 0.8892 | 0.9316 | — | — | **SHIPPED** (`submissions/final.ipynb`) |
| 2026-09-01 | + partner-Ridge as a base FEATURE | 42 | **0.9097** | 0.9084 | 0.9159 | 0.9426 | 0.8583 | 0.9128 | 0.8957 | 0.9338 | — | 502s | biggest single feature-level gain |
| 2026-09-01 | + nested-OOF partner-Ridge feature | 42 | **0.9097** | 0.9085 | 0.9144 | 0.9443 | 0.8557 | 0.9151 | 0.8973 | 0.9329 | — | 1024s | **SHIPPED**; identical mean to the in-sample variant |
| 2026-09-01 | both leaks fixed: clean universe + cross-fitted test weight | 42 | **0.9014** | 0.9085 | 0.9104 | 0.9432 | 0.8333 | 0.9047 | 0.8842 | 0.9256 | — | 282s | **corrected**; CV was inflated +0.008 by the cycle |
| 2026-09-02 | v3: fold-averaged test preds, 10 folds everywhere, two-pass shrunk physics, 3-seed NN | 42 | **0.9015** | 0.9118 | 0.9122 | 0.9406 | 0.8319 | 0.9060 | 0.8819 | 0.9258 | — | 1447s | **SHIPPED**. CV flat vs v2 (0.9014) BY DESIGN — these are transfer fixes and CV cannot measure transfer |
| 2026-09-02 | v4 base: lgbm (3037 feats + dimer aug) | 42 | 0.8721 | | | | | | | | — | 630s | +0.0098 over the 2978-feature build |
| 2026-09-02 | v4 base: cb | 42 | 0.8762 | | | | | | | | — | 962s | |
| 2026-09-02 | v4 base: mtnn 3-seed | 42 | 0.8734 | 0.891 | 0.875 | 0.927 | 0.799 | 0.891 | 0.849 | 0.881 | — | 4577s | |
| 2026-09-02 | **v4 base: periodic GNN** (10 folds, 80 ep, 1 seed) | 42 | **0.8805** | 0.899 | 0.883 | 0.911 | 0.796 | 0.902 | 0.861 | 0.912 | — | 15587s | **best single model**; polyGNN-style periodic graph |
| 2026-09-02 | **v4 FULL: 5 models → stack → 2-pass shrunk physics → partner regression** | 42 | **0.9109** | 0.9178 | 0.9228 | 0.9434 | 0.8531 | 0.9118 | 0.8897 | 0.9378 | — | — | **SHIPPED** (`submissions/final.ipynb`) |
<!-- new runs are inserted above this line by .claude/hooks/log-cv-run.sh -->

## Submission ledger — 3/day, 2 final picks, deadline 3 Sep 2026

| date | config / notebook | local CV | public LB | pinned? | shared with hosts? |
|---|---|---|---|---|---|
| earlier | `submissions/aisehack3-1.ipynb` | OOF ~0.903 | **0.883** | ? | ? |
| 2026-09-01 | `submissions/final.ipynb` (buggy: cyclic universe + in-sample test weight) | 0.9097 | **0.860** | | |
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
- **Iterated physics refinement leaks catastrophically unless masked per fold.**
  Refining the predicted property table by belief propagation over the relation
  graph looked worth **+0.042 mean R² (0.893 → 0.935)**. It was entirely fake:
  `eps` is refined *from* `nc`, and `nc` is then predicted *from* the refined
  `eps`, so a validation polymer's own measured label flowed back into its own
  prediction through a two-hop loop. With the mask in place (each (target, fold)
  redoes the refinement with that fold's true target values replaced by their
  out-of-fold predictions) the real gain is **+0.001**. Keep the mask; the
  iteration itself is close to worthless here. This is the same class of error
  that most plausibly explains the 0.903 OOF → 0.883 LB gap on the team's
  earlier notebook.
- **A Ridge stack over a single base model makes things slightly worse**
  (0.8645 → 0.8622 on lgbm alone). The meta-learner needs ≥2 decorrelated inputs
  before it earns its variance.
- **The partner block is worth more as one FITTED column than as raw columns.**
  The trees already receive every `true_<prop>` value, but an axis-aligned split
  approximates a *linear combination* of them badly and the DFT block is close to
  linear in exactly that way. Ridge-fit the combination on the training fold and
  hand the model the single fitted value. Measured honestly through the whole
  pipeline: **0.9070 → 0.9097 (+0.0027)**.
- **CORRECTION — an earlier "+0.069 on eps" for this feature was leaky.** That
  first test built the Ridge over the *prediction-filled* partner frame
  (`oof.partner_frame`), where a missing partner is filled with a base-model
  prediction. 52-53% of partner cells for eps/nc/ei are prediction-filled, and
  every one of those predictions came from a model that receives `true_<target>`
  as a feature — so the row's own label returns through one hop. The shipped
  implementation deliberately uses the **mean-filled** block from
  `partners.build()`, which is rebuilt per fold with the validation rows removed.
  Same shape of mistake as the relation-graph refinement: any partner value that
  is itself a model output can close a cycle. Check the provenance of every cell
  before using it as an input.
- **Nested-OOF for that feature is principled but does not move the score.** An
  in-sample Ridge fit is unrealistically accurate on the rows it was fitted on,
  so the tree over-trusts a column that is weaker at inference — the classic
  target-encoding failure. A nested 5-fold fixes it and is worth +0.002 to +0.004
  at the *base-model* level (eps +0.0023, nc +0.0042, ei +0.0027), but through
  the stack and physics the final score is **0.9097 either way** — identical to
  four decimals. Kept because it is the correct construction, not because it
  helps. A base-model gain that survives the stack is the exception, not the
  rule: the stack is already correcting much of what it fixes.
- **Repeat-unit (dimer) augmentation is the biggest accuracy lever we had missed.**
  `*CC*` and `*CCCC*` are the same polymer with the same property value, so the
  dimer of a training row is a new view of a known label. Augmented rows enter
  TRAINING folds only. Measured on lgbm, 10 folds:

      eps 0.7882 -> 0.8230  (+0.0349)      ei  0.8115 -> 0.8196  (+0.0081)
      nc  0.8607 -> 0.8672  (+0.0065)      eea 0.8769 -> 0.8788  (+0.0019)
      egb 0.9042 -> 0.9061  (+0.0019)
      mean over the 5 small targets +0.0106  =  +0.0076 on the 7-target mean

  I had explicitly dismissed this. I measured that the data contains no oligomer
  duplicates and concluded invariance was "a rubric deliverable, not points".
  That was right about DEduplication and missed the inverse entirely: you can
  CREATE the variants as training data. Same polymer, same label, different
  features, free rows exactly where the labels are scarce.
- **Intensive features are a repetition-invariance choice, not just more columns.**
  Fraction of feature values unchanged when a repeat unit is dimerised:
  element fractions 100%, side-chain ratio 100%, intensive twins 87.6%, against
  62.8% for raw RDKit descriptors. MolWt and HeavyAtomCount double exactly
  (ratio 2.000) as the control. Extensive quantities scale with the unit count;
  intensive ones do not.
- **The graph readout IS the invariance guarantee.** On an untrained periodic-graph
  net, prediction deviation under dimerisation: mean/max readout ~1e-4 median,
  sum readout ~0.7 -- four orders of magnitude. Atom-ordering invariance is EXACT
  (0.00e+00) for both. Note this is measured NEAR-invariance for repeat count, not
  a proof: ~97% of polymers give structurally identical node-environment
  distributions, and the residual 3% are genuine mismatches. Do not overclaim it.
- **We refit on 100% of the data for test predictions; they average the fold
  models. That was the structural bug.** `per_property_oof` trained K fold models
  to produce the OOF, threw them away, then fit ONE fresh model on all rows for
  the test prediction. So the stack's Ridge coefficients and the physics weights
  were calibrated on columns from (K-1)/K-trained models and then fed a column
  from a 100%-trained model — different objects. It also discarded free bagging
  on 3 of our 4 base models (only the NN averaged folds). Their notebook averages
  ~80 fold models per test row and refits nothing. Fixed in v3.
- **Shrink every fitted blend weight to 0.75 of its value.** An argmax over a
  21-point grid chosen on 50-135 rows is biased high by construction;
  cross-fitting averages the noise but not the bias. Measured FREE locally
  (0.9014 → 0.9020) while cutting test-side perturbation 25-33%.
- **Per-level physics staging bought nothing.** One weight per target scored
  0.9006 vs 0.9005 for the staged version — identical, with a third of the free
  parameters and ~30% less test movement. Replaced with a two-pass split
  (all sources measured / not).
- **CV cannot measure a transfer fix.** v3 scores 0.9015 against v2's 0.9014.
  That is the expected and correct outcome: fold-averaging, shrinkage and
  parameter reduction change how well the pipeline generalises, not how well it
  fits the validation set. Do not read a flat CV as "no improvement" here.
- **Do NOT copy their per-fold top-500 feature selection** — measured, it LOSES
  0.0046 on our small targets. And note all three of their notebooks early-stop
  GBDTs on the fold's own validation rows, which inflates their 0.903 OOF; the
  real model-quality gap to their 0.883 LB is wider than the headline suggests.
- **THE BIG ONE — the partner universe was cyclic, and it cost a submission.**
  `partner_frame()` fills a *missing* partner with a model prediction. But those
  predictions come from base models that receive `true_<other>` as features, so
  the value filled into `P[nc][p]` is a function of p's own `eps` label. The
  physics blend and the partner regression then predict `eps` from it. One hop,
  and invisible because every individual stage looked correctly cross-fitted.
  Rebuilding the universe from a partner-FREE pass: **0.9097 → 0.9014 (−0.008)**,
  and the partner-regression stage's gain collapses from +0.008 to +0.003 on eps,
  +0.003 on nc, and NEGATIVE on egb/ei. That stage was almost entirely leak.
- **The test-side blend weight was tuned in-sample — a pure CV-to-LB gap
  generator.** `partner_regression` re-fitted a RidgeCV on all rows, predicted
  those same rows, and tuned the blend weight against that in-sample estimate,
  then applied the weight to genuinely out-of-sample test predictions. Measured
  inflation: eea 0.003 → 0.250, ei 0.290 → 0.550, eps 0.457 → 0.650. It degrades
  ONLY the test side, so no amount of CV would ever reveal it. Both stages now
  reuse the cross-fitted fold weights.
- **Submitted 0.9097 local → 0.860 public LB.** Worse than the team's existing
  0.883. The two bugs above account for part of it; roughly 0.02 of gap remains
  unexplained versus that notebook's own −0.020 CV-to-LB offset. Do not treat a
  local number from this pipeline as an LB estimate without that offset.
- **Turn the relation-graph refinement OFF (`n_rounds=0`).** Once the per-fold
  mask is in place it is not merely small, it is slightly negative on the
  four-model stack: 0.9047 with no refinement vs 0.9040 with two rounds. It only
  looked useful (+0.001) against a single weak base model. Keep the code — the
  masking logic is what documents *why* the naive version was fake — but ship it
  disabled. Simpler and measured no worse.
- **The physics blend-weight grid granularity does not matter.** 21 steps, 51
  steps and a grid extended past 1.0 all give an identical 0.9047. The weight is
  not being pinned by grid resolution.
- **The SMILES CNN is not worth its runtime.** Standalone 0.8206 (1276 s/seed).
  Added to lgbm+xgb+cb+mtnn it moves the final score 0.9040 → 0.9045: **+0.0005
  for ~42 min locally and ~2 h of Kaggle CPU at two seeds.** Dropped. It is a
  genuinely decorrelated view (it reads the SMILES string, not descriptors), but
  decorrelation at 0.82 does not buy enough against a 0.90 stack.
- **Cheap diverse models (kNN on Morgan bits, ExtraTrees, Ridge) do not help the
  stack.** Standalone with partner features: knn 0.7412 (29s), ridge 0.5939 (4s).
  Adding them to lgbm+xgb+cb+mtnn moved the final score 0.9040 → 0.9034 (knn) and
  → 0.9037 (knn+ridge). Diversity only pays when the added model is *also* strong;
  a 0.74 model forces the meta-learner to spend capacity down-weighting it.
  `src/models/simple.py` is kept for reference but is not in the pipeline.
- **A fourth booster is nearly worthless.** xgb on top of lgbm+cb was +0.0001.
  The three GBDTs correlate at ~0.99; the multi-task NN is what actually moved
  the stack (0.8713 → 0.8950), because it is wrong in different places.
- **`submissions/round-3-aisehack.ipynb` (v2) is not a safe default.** Unscored,
  drops ingest canonicalisation, fits one physics weight across two populations,
  and very likely exceeds the Kaggle session limit. Score it before trusting it.
