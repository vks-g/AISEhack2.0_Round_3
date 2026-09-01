---
name: kaggle-export
description: Flatten the local src/ pipeline into one self-contained Kaggle notebook that trains from scratch in a single run. Use when preparing or updating a submission notebook.
---

# Export src/ to a Kaggle notebook

The competition is notebook-only (§6.2.2). A prediction file not backed by a
compliant notebook is invalidated, and after the deadline the hosts execute the
**pinned** version and require it to reproduce the submitted score exactly (§7.2).

For validating the *output*, use the `submission-check` skill. This skill is
about producing the notebook.

## Cell order

1. `!pip install rdkit -q` — the only install; everything else is preinstalled
2. imports, then a seed block that sets `random`, `numpy`, `torch` and prints
   every seed it set
3. `src/smiles_utils.py` — canonicalisation and rewritings
4. `src/metric.py` — targets and R²
5. `src/features.py` — **with `use_cache=False`**; strip the `.cache` disk paths
6. `src/partners.py` + the leakage self-test, which must `raise` on failure
7. `src/physics.py`
8. the chosen `src/configs/<name>.py`, inlined
9. full fit on all of train, then inference on test
10. `submission.csv` write + `head()` + the per-target range print
11. explainability section (`src/explain.py` logic)
12. invariance certificate (`src/invariance.py` logic)

No `from src import ...` — Kaggle has no repo. Everything literal.

## Rules the export must not break

- **Strip every disk cache.** `src/features.py` memoises to `.cache/` locally so
  CV iterates in seconds. That directory must not exist in the notebook: §6.2.4
  bans shipping cached feature files. Featurising all 12,345 molecules from
  scratch takes ~25 s on 9 processes — it is not a bottleneck, so there is no
  reason to cache in the notebook at all.
- **No wall-clock branching.** Anything of the form "train until N hours elapse"
  makes the run non-reproducible and voids the submission under §7.2. Fixed epoch
  counts only.
- **`DATA_DIR` must resolve to the competition dataset alone.** Do not
  `glob('/kaggle/input/*')` — it can silently pick up an attached dataset, which
  is a §6.2.1 disqualification. Name the path.
- **Nothing read that this run did not write.** No pickles, no `.pt`, no
  embeddings, no `archive/`, no `Round 2 /`.
- Seeds set and printed. Output named exactly `submission.csv`.

## Runtime budget

Budget the full fit locally first and write the measured number into the notebook
header. For reference on this machine: featurisation ~25 s, and the `lgbm` config
is 377 s for a full 10/15-fold CV, so a single full fit is far less.

`submissions/round-3-aisehack.ipynb` (the unscored "v2") adds an MPNN, a 5-seed
CNN, per-fold feature selection and a 2-hour SSL branch. Its total runtime very
likely exceeds a single Kaggle session, which produces **no submission at all**.
Time it before trusting it.

## The two notebooks already in this repo

- `submissions/aisehack3-1.ipynb` — **the incumbent, public LB 0.883**, OOF ~0.903.
- `submissions/round-3-aisehack.ipynb` — "v2", never scored. Its own markdown
  admits it. It drops ingest canonicalisation (weakening the invariance claim
  from exact to approximate), replaces the two-pass physics blend with a single
  coalesced pass fitted across two populations, adds a hard `import shap` that v1
  had deliberately engineered away, and still warns `len(test_df) != 4497` on
  every correct run. Treat it as a candidate to be scored, not as the incumbent.

Score anything through `src.cv` before it gets a slot.
