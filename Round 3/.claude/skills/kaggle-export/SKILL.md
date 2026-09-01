---
name: kaggle-export
description: Generate the Kaggle submission notebook from src/ and prove it runs end to end. Use when preparing or updating a submission notebook.
---

# Export src/ to a Kaggle notebook

The notebook is **generated, never hand-edited**, so it cannot drift from the
code that was scored:

```bash
cd "Round 3"
./.venv/bin/python -m src.export_notebook --models lgbm,xgb,cb,mtnn,cnn \
    --out submissions/final.ipynb
./.venv/bin/python scripts/execute_notebook.py submissions/final.ipynb
./.venv/bin/python -m src.check_submission /tmp/nbrun/submission.csv
```

Edit `src/`, then re-export. Never patch the .ipynb by hand.

## What the generator has to handle

Flattening a package into one namespace is where this breaks silently. Each of
these was a real bug caught by executing the notebook:

- **Intra-package imports** are stripped. An import that was the *body* of a
  `try:` leaves an empty block, so those are replaced with `pass`, not deleted.
- **Name collisions.** `models/mtnn.py` and `models/cnn.py` both export
  `oof_and_test`; inlined unchanged, only the second survives and the notebook
  silently trains the CNN twice. They are renamed on inlining. Same for the three
  modules that each define `report`.
- **Cross-module helpers.** `models/trees.py` imports `params_for` from
  `configs/lgbm.py`; that function is inlined under the alias trees.py expects,
  or the notebook dies with `NameError` at the first model.
- **The disk cache is removed, not neutered.** `features.py` memoises to
  `.cache/` locally. Leaving even the dead `np.load`/`np.savez` bodies in means a
  host grepping for artifact I/O gets hits on code that never runs.

## Two platform traps that only appear in the notebook

**Import LightGBM before PyTorch.** They bundle separate OpenMP runtimes. On
macOS, initialising torch's first makes LightGBM segfault later inside
`lightgbm/basic.py __init_from_np2d` — a bare SIGSEGV, no Python traceback,
normal memory use. Measured: no-torch survives, lightgbm→torch survives,
torch→lightgbm crashes. Linux (so Kaggle) resolves both to one libgomp and is
unaffected, but the seed cell imports the boosters first so the notebook runs
anywhere. Do not reorder it.

When something dies with no traceback, run it under `-X faulthandler` — that is
what produces the crash site. And read the *signal*: a wrapper that reports the
child's negative return code distinguishes SIGKILL (memory) from SIGSEGV (a
library fault) in one run. Free-page counts on macOS are not available memory and
will send you the wrong way.

**Data discovery must search, but must not adopt.** `DATA_DIR` walks
`/kaggle/input` to depth 3 and accepts a directory only if it holds **both**
`train.csv` and `test.csv`. The pairing is what prevents picking up an unrelated
attached dataset (§6.2.1) — not the absence of a walk. A one-level scan failed to
find the competition's own data. Every match is printed and the selection named,
and failure raises with the actual tree so one run diagnoses it.

## Rules the export must not break

- **Nothing read that this run did not write.** No checkpoint import, no feature
  pickle, no `torch.load`. `submissions/pie-net-v3-checkpoint-run-2.ipynb` globs
  `/kaggle/input/**/ckpt/manifest.json` to adopt a previous run's output — that
  is §6.2.4, §6.2.2 and §7.2 at once, and its own docstring says "attach the
  previous version's output and run again".
- **No wall-clock branching.** `round-3-aisehack.ipynb` trains
  `while elapsed < SSL_MAX_HOURS`, which cannot reproduce a pinned score under
  §7.2. Every loop here is a fixed fold/epoch/iteration count, so a slower
  machine gives the *same* answer, only later.
- **`DATA_DIR` from an explicit candidate list**, then at most one shallow scan of
  the immediate children of `/kaggle/input`. Never a recursive glob — that can
  adopt an attached dataset (§6.2.1).
- Seeds set and printed. Output named exactly `submission.csv`.

## Verify before submitting — all four

```bash
# 1. structure: syntax, duplicate defs, forbidden patterns
./.venv/bin/python -c "import json,ast,re,collections; \
nb=json.load(open('submissions/final.ipynb')); \
cc=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']; \
w=chr(10).join(cc); \
print({b:w.count(b) for b in ['torch.load','pickle.load','np.load(','MAX_HOURS','from_pretrained','torch.hub','from src'] if w.count(b)} or 'clean')"

# 2. it actually runs, and writes a submission
./.venv/bin/python scripts/execute_notebook.py submissions/final.ipynb

# 3. the submission is valid
./.venv/bin/python -m src.check_submission /tmp/nbrun/submission.csv

# 4. the notebook's own asserts passed (invariance certificate + compliance audit)
```

The notebook carries two `assert`s on purpose: permutational invariance must be
exact, and the compliance audit must find no other attached dataset. If either
fails the notebook fails loudly rather than producing a quietly wrong submission.

## Runtime

Measured locally on 11 cores, per full OOF pass: featurisation 25 s, LightGBM
343 s, XGBoost 232 s, CatBoost 486 s, multi-task NN 1098 s per seed, SMILES CNN
~1100 s per seed, stack+physics ~10 s. Kaggle CPU gives 4 cores, so expect ~2.5x
— on the order of 3-4 hours with two NN seeds, inside the 12-hour limit. A GPU
session cuts the two neural models sharply.

`NN_SEEDS` is the runtime dial. Drop it to `[42]` if a session is at risk; the
cost is the seed-averaging benefit, not correctness.
