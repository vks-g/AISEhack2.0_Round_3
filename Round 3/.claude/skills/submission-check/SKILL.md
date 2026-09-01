---
name: submission-check
description: Validate a submission.csv against test.csv and the competition rules before spending one of the three daily upload slots. Use before every single submission.
---

# Before you upload

```bash
cd "Round 3" && ./.venv/bin/python -m src.check_submission submission.csv
```

Exits 0 on pass, 1 on fail, and prints a per-target range table. Every threshold
is computed from `test.csv` / `train.csv` at run time — nothing is hard-coded,
because the hard-coded row count this repo used to carry was wrong.

## The row count trap

`test.csv` has **4940 rows**. It has 4497 *unique raw SMILES*, and the Kaggle
data page's "4,497 data points" refers to molecules. A submission with 4497 rows
is rejected. `submissions/round-3-aisehack.ipynb` still carries a
`if len(test_df) != 4497` warning that fires on every correct run — ignore it,
or fix it.

`sample_submission.csv` has only **10 rows**. It illustrates the format; it is
not a template to join against.

## What the checker enforces

- columns are exactly `id,target`
- row count equals `len(test.csv)`
- the id set matches test.csv exactly, no duplicates, none missing
- all values numeric and finite
- per-`target_type` predictions fall inside the observed train range plus 25%
- no target_type has constant predictions (catches a broken target mapping)

Per-target train ranges, for eyeballing: tg [-109.8, 495.0] · egc [0.02, 9.86] ·
egb [0.51, 10.11] · eps [2.61, 9.09] · **nc [1.56, 2.76]** · ei [4.03, 9.84] ·
eea [0.39, 5.14]. Note nc — it is 1.56–2.76, not the "1.5–1.7" this repo used to
claim.

## Disqualification checklist — all must hold

These are rule violations, not score problems. Any one of them voids the
submission regardless of leaderboard position (rules §6.2.x, §7.x).

- [ ] Output file named exactly `submission.csv`
- [ ] **Zero external datasets attached** to the notebook. Only the official
      competition data. `archive/`, `Round 2 /`, and any locally produced file
      are external data.
- [ ] **Zero uploaded weights, checkpoints, embeddings or cached features.**
      No `from_pretrained`, no `torch.hub`, no HuggingFace, no `.pt`/`.pkl`
      loaded from anywhere the notebook did not itself write in this run.
- [ ] Every stage runs inside one notebook execution: load → preprocess → split
      → train → infer → write. No manual steps.
- [ ] Every seed set and printed.
- [ ] **No wall-clock-dependent branching.** A time budget (`SSL_MAX_HOURS`, an
      elapsed-time early stop) makes the run non-reproducible, and §7.2 voids a
      submission whose pinned notebook does not reproduce the submitted score.
- [ ] No `glob('/kaggle/input/*')` that could silently pick up an attached
      dataset.
- [ ] Notebook shared with view access to all five hosts: Rohit Batra IITM,
      Rahulsundar, LaksmanN, VIJITH P, shreyasri0301.
- [ ] Submission description links the notebook.
- [ ] The **default/pinned** notebook version is the one that produced this score.

Quick grep for the banned patterns:

```bash
grep -nE 'from_pretrained|torch\.hub|huggingface|transformers|read_pickle|joblib\.load|/kaggle/input/\*' submissions/<notebook>.ipynb
```

## Slot discipline

3 per day, 2 final picks, and the deadline is 3 Sep 2026. Do not spend a slot on
a config whose local CV has not beaten the current best by more than 2x the fold
noise. Pick two finals that differ in approach rather than two seeds of one
model — the public LB is only 37% of the test set.
