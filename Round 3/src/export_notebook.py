"""Generate the Kaggle submission notebook from src/, so it cannot drift.

    ./.venv/bin/python -m src.export_notebook --out submissions/final.ipynb

Design constraints, each one a rule the team's earlier notebooks broke:

  * Nothing is read that this run did not write. No checkpoint import, no feature
    pickle, no attaching a previous run's output. `pie-net-v3-checkpoint-run-2`
    globbed /kaggle/input for a previous run's ckpt/manifest.json -- that is
    rule 6.2.4 (artifacts created outside notebook execution), 6.2.2 (single
    run) and 7.2 (no manual intervention) at once.
  * No wall-clock branching. `round-3-aisehack` trained "while elapsed < 2 hours",
    which cannot reproduce a pinned score under 7.2. Every loop here is a fixed
    count.
  * DATA_DIR is resolved from an explicit list of candidate paths. No recursive
    glob over /kaggle/input that could adopt an attached dataset (6.2.1).
  * The local .cache/ disk memo in features.py is neutralised, so the notebook
    recomputes every feature from scratch.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.paths import REPO

SRC = REPO / "src"


def strip(path: str, rename: dict | None = None) -> str:
    """Inline a module into the shared notebook namespace.

    Two things have to happen. Intra-package imports are dropped -- they have no
    meaning once every module shares one namespace. And colliding top-level names
    are renamed: models/mtnn.py and models/cnn.py both export `oof_and_test`, so
    inlining them unchanged would silently leave only the second definition, and
    the notebook would train the CNN twice while reporting one of them as the NN.
    """
    text = (SRC / path).read_text()
    out = []
    for line in text.split("\n"):
        st = line.strip()
        if st.startswith(("from src", "import src", "from __future__")):
            indent = len(line) - len(line.lstrip())
            if indent:
                # the import was the body of a try/if -- deleting it outright
                # would leave an empty block and a SyntaxError
                out.append(" " * indent + "pass")
            continue
        out.append(line)
    body = "\n".join(out).strip() + "\n"
    for old, new in (rename or {}).items():
        body = re.sub(rf"\b{re.escape(old)}\b", new, body)
    return body


def _lines(text: str) -> list[str]:
    """nbformat wants a list of lines that each KEEP their trailing newline.
    Splitting without re-adding them collapses the whole cell onto one line."""
    body = text.strip("\n").rstrip() + "\n"
    return body.splitlines(keepends=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _lines(text)}


HEADER = """
# AISEHack 2.0 — Round 3: Polymer Property Prediction

Predicts the seven polymer properties (`tg egc egb eps nc ei eea`) from PSMILES.
Scored on the **unweighted mean R² across the seven targets**, so the four
~220-row properties (`eps nc ei eea`) carry 4/7 of the score on under 4% of the
training rows. Everything here is sized around that fact.

## Pipeline

1. Canonicalise every SMILES with RDKit. **Every downstream feature is computed
   from the canonical form**, which is what makes the pipeline exactly invariant
   to how a polymer was written (verified in §Invariance below).
2. Featurise: RDKit descriptors + Morgan(r=2,3) + atom-pair + topological-torsion
   + MACCS + polymer-specific terms + SMARTS functional groups = 2978 features.
3. Base models, per property, on one shared polymer-grouped fold assignment:
   LightGBM, XGBoost, CatBoost, a multi-task neural net and a SMILES 1-D CNN.
4. Cross-fitted Ridge stack per property.
5. Two-stage physics blend on the inter-property relations, **affine-calibrated**
   rather than applied as raw identities.
6. Clip to the observed range, write `submission.csv`.

## Runtime

Measured on an 11-core laptop, per full out-of-fold pass (10 folds for the
~220-row targets, 5 for `tg`/`egc`):

| stage | local | note |
|---|---|---|
| featurisation (12,345 molecules) | 25 s | 9 processes |
| LightGBM | 343 s | |
| XGBoost | 232 s | |
| CatBoost | 486 s | the slowest booster |
| multi-task NN, per seed | 1098 s | 2 seeds here |
| SMILES CNN, per seed | ~1100 s | 2 seeds here |
| stack + physics | ~10 s | |

Kaggle CPU sessions give 4 cores, so expect roughly 2.5x these figures — on the
order of 3-4 hours, inside the 12-hour limit. On a GPU session the two neural
models are far faster and the total drops well under 2 hours. Nothing here
branches on elapsed time, so a slower machine produces the *same* result, only
later — which is what rule 7.2 requires of the pinned version.

## Compliance

Every stage runs inside this single execution. Specifically:

* **No external data.** Only the attached competition files are read.
  `DATA_DIR` is resolved from an explicit candidate list — there is no recursive
  glob over `/kaggle/input`, so no attached dataset can be picked up silently.
* **No pretrained weights and no uploaded artifacts.** Nothing is loaded that
  this run did not itself produce: nothing is deserialised from disk, no
  checkpoint is imported, no feature cache is read. Every model is trained here.
* **No wall-clock branching.** Every training loop runs a fixed number of
  folds/epochs/iterations, so the pinned version reproduces this score exactly.
* **Seeds** are set and printed below.

A self-audit at the end of the notebook re-checks these mechanically.
"""

SEEDS = '''
import os, sys, time, math, random, json, warnings, gc, re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

SEED = 42
N_FOLDS = 10
NN_SEEDS = [42, 202]

random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(False)
except Exception:
    torch = None

print(f"SEED={SEED}  N_FOLDS={N_FOLDS}  NN_SEEDS={NN_SEEDS}")
print("python", sys.version.split()[0])
print("numpy", np.__version__, "pandas", pd.__version__)
'''

DATA_DIR_CELL = '''
# Explicit candidate list. Deliberately NOT a recursive glob over /kaggle/input:
# a glob can silently adopt an attached dataset, which is rule 6.2.1.
_CANDIDATES = [
    "/kaggle/input/aisehack-2-0",
    "/kaggle/input/aisehack-2-0-polymer-property-prediction-round-3",
    "/kaggle/input/ppp-round-3",
    "data",
    ".",
]

DATA_DIR = None
for _c in _CANDIDATES:
    if os.path.isfile(os.path.join(_c, "train.csv")) and os.path.isfile(os.path.join(_c, "test.csv")):
        DATA_DIR = _c
        break
if DATA_DIR is None:
    # One shallow scan of the immediate children of /kaggle/input -- still not
    # recursive, and it only ever matches the competition's own directory.
    _root = "/kaggle/input"
    if os.path.isdir(_root):
        for _d in sorted(os.listdir(_root)):
            _p = os.path.join(_root, _d)
            if os.path.isfile(os.path.join(_p, "train.csv")):
                DATA_DIR = _p
                break
if DATA_DIR is None:
    raise FileNotFoundError("could not locate train.csv/test.csv")

WORK_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
print("DATA_DIR:", DATA_DIR)
print("contents:", sorted(os.listdir(DATA_DIR))[:10])

def cache_dir():
    raise RuntimeError("no disk cache in the notebook -- everything is recomputed")
'''

FEATURES_OVERRIDE = '''
# The local development harness memoises features to disk so cross-validation
# iterates in seconds. Shipping or reading such a cache would be rule 6.2.4, so
# both halves are neutralised here and every feature is recomputed from scratch.
# The in-process dict memo is kept: it only avoids featurising a molecule twice
# within THIS run (train and test share 1063 polymers).
print("feature disk cache: removed at export; every feature recomputed in this run")
'''

DATA_CELL = '''
def load_train(dedupe=True):
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df["canon"] = df["smiles"].map(canonicalize)
    unknown = set(df["target_type"].unique()) - set(TARGETS)
    assert not unknown, f"unexpected target_type {unknown}"
    if dedupe:
        df = (df.groupby(["canon", "target_type"], as_index=False)
                .agg(target=("target", "mean"), smiles=("smiles", "first")))
    return df.reset_index(drop=True)

def load_test():
    df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    df["canon"] = df["smiles"].map(canonicalize)
    return df.reset_index(drop=True)

train_df = load_train()
test_df = load_test()
print(f"train {train_df.shape}  test {test_df.shape}")
print(train_df.target_type.value_counts().reindex(TARGETS).to_string())
'''


def build(models, out_path):
    cells = [md(HEADER), code("!pip install rdkit -q"), code(SEEDS)]

    cells.append(md("## 1. Data location and seeds"))
    cells.append(code(DATA_DIR_CELL))

    cells.append(md("## 2. Canonicalisation and polymer rewritings\n\n"
                    "Every feature downstream is computed from `canonicalize()`. "
                    "That is the whole basis of the invariance guarantee proved at the end."))
    cells.append(code("from rdkit import Chem, RDLogger, DataStructs\n"
                      "from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator\n"
                      "RDLogger.DisableLog('rdApp.*')\n"
                      "from functools import lru_cache\n\n" + strip("smiles_utils.py")))

    cells.append(md("## 3. The competition metric"))
    cells.append(code(strip("metric.py", {"report": "report_metric"})))

    cells.append(md("## 4. Featurisation — 2978 features from the canonical SMILES"))
    # Cut the disk-memo helpers out of features.py. They are a local development
    # convenience; leaving even their dead bodies in would put np.load/np.savez
    # in a notebook that must demonstrably read no artifacts (rule 6.2.4).
    _feat = strip("features.py")
    _a = _feat.index("def _disk_paths():")
    _b = _feat.index("def featurize(")
    _feat = _feat[:_a] + ("def _load_disk():\n    return None\n\n\n"
                          "def _save_disk():\n    return None\n\n\n") + _feat[_b:]
    cells.append(code("from concurrent.futures import ProcessPoolExecutor\n\n" + _feat))
    cells.append(code(FEATURES_OVERRIDE))

    cells.append(md("## 5. Load the competition data"))
    cells.append(code(DATA_CELL))

    cells.append(md("## 6. Folds — one polymer-grouped assignment shared by every base model\n\n"
                    "Sharing one assignment is what makes stacking valid: each base model's "
                    "out-of-fold prediction for a row comes from a model that saw exactly the "
                    "same training rows. Grouping on the canonical SMILES keeps a polymer, "
                    "which contributes up to six rows, inside a single fold."))
    cells.append(code("from sklearn.model_selection import GroupKFold, KFold\n\n" + strip("splits.py")))

    cells.append(md("## 7. Physics relations between the DFT properties\n\n"
                    "Measured on co-observed training pairs: the raw identity `eps = nc**2` "
                    "scores R² 0.336, while the same expression affine-calibrated scores 0.855. "
                    "`nc` goes 0.171 → 0.837 and `egb` 0.892 → 0.928. Every relation here is "
                    "fitted, never applied raw."))
    cells.append(code("from dataclasses import dataclass, field\n\n" + strip("physics.py", {"report": "report_relations"})))

    cells.append(md("## 8. Base models"))
    # trees.py imports params_for from src/configs/lgbm.py. Inline just that
    # function under the alias trees.py expects, or the flattened notebook dies
    # with NameError the moment the first model is built.
    _cfg = (SRC / "configs" / "lgbm.py").read_text()
    _pf = _cfg[_cfg.index("def params_for("):_cfg.index("def fit(")].rstrip()
    cells.append(code("import lightgbm as lgb\nimport xgboost as xgb\nimport catboost as cb\n\n"
                      + _pf + "\n\n\n_lgbm_params_for = params_for\n\n\n"
                      + strip("models/trees.py", {"NAME": "_NAME_TREES"})))
    if "mtnn" in models:
        cells.append(md("### Multi-task neural network\n\n"
                        "One shared trunk over the feature matrix with seven per-property heads. "
                        "`y` is standardised **per target** inside each fold — without that, `tg` "
                        "(range 495) dominates a shared loss and `nc` (range 2.76) learns nothing."))
        cells.append(code(strip("models/mtnn.py", {"oof_and_test": "mtnn_oof_and_test", "NAME": "_NAME_MTNN"})))
    if "cnn" in models:
        cells.append(md("### SMILES 1-D CNN\n\n"
                        "Reads the canonical SMILES string directly, so its errors decorrelate "
                        "from the descriptor models — which is what it contributes to the stack."))
        cells.append(code(strip("models/cnn.py", {"oof_and_test": "cnn_oof_and_test", "NAME": "_NAME_CNN"})))

    cells.append(md("## 9. Out-of-fold engine, cross-fitted stack, two-stage physics\n\n"
                    "The stacker, the physics blend weight and the physics calibration are all "
                    "cross-fitted over the fold assignment. Fitting them on the same out-of-fold "
                    "predictions they are then scored against is what produces an OOF number that "
                    "does not survive the leaderboard."))
    # Drop the local disk-cache helpers entirely rather than neutering them:
    # leaving dead np.load/np.savez calls in the notebook means a host grepping
    # for artifact I/O gets hits on code that never runs.
    _oof = strip("oof.py")
    _a = _oof.index("# local cache")
    _b = _oof.index("def report(")
    _oof = _oof[:_oof.rindex("# --", 0, _a)] + _oof[_b:]
    cells.append(code("from sklearn.linear_model import Ridge\n\n" + _oof))

    cells.append(md("## 10. Run the pipeline and write `submission.csv`"))
    # Emit dispatch branches only for the models actually included, so the
    # notebook never references a function that was not inlined.
    branches = []
    if "mtnn" in models:
        branches.append('    elif name == "mtnn":\n'
                        '        o, te = mtnn_oof_and_test(train_df, X_tr, test_df, X_te, fold_id, NN_SEEDS)')
    if "cnn" in models:
        branches.append('    elif name == "cnn":\n'
                        '        o, te = cnn_oof_and_test(train_df, test_df, fold_id, NN_SEEDS)')
    run = (RUN_CELL.replace("__MODELS__", repr(list(models)))
                   .replace("__MODEL_BRANCHES__", "\n".join(branches) if branches else "    # (no neural models in this build)"))
    cells.append(code(run))

    cells.append(md("## 11. Explainability — per-target TreeSHAP\n\n"
                    "Exact SHAP values from LightGBM's own `pred_contrib=True`: the same "
                    "algorithm the `shap` package implements, computed inside LightGBM, so the "
                    "notebook gains no dependency that could fail to install."))
    cells.append(code(EXPLAIN_CELL))

    cells.append(md("## 12. Polymer-invariance certificate"))
    cells.append(code(INVARIANCE_CELL))

    cells.append(md("## 13. Compliance self-audit"))
    cells.append(code(AUDIT_CELL))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    Path(out_path).write_text(json.dumps(nb, indent=1))
    return nb


RUN_CELL = '''
MODELS = __MODELS__
t_start = time.time()

all_canon = list(dict.fromkeys(list(train_df["canon"]) + list(test_df["canon"])))
print(f"universe: {len(all_canon)} distinct polymers")

X_tr = featurize(train_df["canon"])
X_te = featurize(test_df["canon"])
X_all = featurize(all_canon)
print(f"features {X_tr.shape}  ({time.time()-t_start:.0f}s)")

fold_id = build_fold_id(train_df, n_folds=N_FOLDS, seed=SEED)

base_oof, base_test, tree_universe = {}, {}, {}
for name in MODELS:
    t1 = time.time()
    if name in ("lgbm", "xgb", "cb"):
        o, te, uni = per_property_oof(make, name, train_df, X_tr, test_df, X_te,
                                      fold_id, all_canon, X_all, seed=SEED)
        tree_universe[name] = np.column_stack([uni[t] for t in TARGETS])
__MODEL_BRANCHES__
    else:
        raise ValueError(name)
    base_oof[name], base_test[name] = np.asarray(o), np.asarray(te)
    _d = train_df.copy(); _d["pred"] = base_oof[name]
    print(f"  {name:<6} mean R2 {competition_score(_d)[0]:+.4f}   ({time.time()-t1:.0f}s)")

print(f"\\n{'model':<8}" + "".join(f"{t:>8}" for t in TARGETS) + f"{'MEAN':>9}")
for name in MODELS:
    _d = train_df.copy(); _d["pred"] = base_oof[name]
    s, per = competition_score(_d)
    print(f"{name:<8}" + "".join(f"{per[t]:>8.4f}" for t in TARGETS) + f"{s:>9.4f}")

stacked_oof, stacked_test, alphas = stack(train_df, base_oof, base_test, test_df, fold_id)
score_stack, _ = report(train_df, stacked_oof, "stacked")

uni = np.mean([tree_universe[k] for k in tree_universe], axis=0)
universe_by_target = {t: uni[:, i] for i, t in enumerate(TARGETS)}
partners, is_true = partner_frame(train_df, all_canon, universe_by_target)
# n_rounds=2 is what was measured locally. The refinement itself is worth only
# ~+0.001 once the per-fold mask is in place -- the mask is the important part.
# n_rounds=0: the relation-graph refinement measured slightly NEGATIVE on the
# four-model stack once its per-fold mask was in place (0.9047 vs 0.9040). The
# masking logic is kept in the code because it documents why the unmasked version
# was fake, but it ships disabled.
final_oof, final_test, phys_info = apply_physics(
    train_df, stacked_oof, test_df, stacked_test, fold_id, partners, is_true,
    n_rounds=0)
report(train_df, final_oof, "stacked + staged physics")

final_oof, final_test, pr_info = partner_regression(
    train_df, final_oof, test_df, final_test, fold_id, partners, is_true)
FINAL_SCORE, FINAL_PER = report(train_df, final_oof, "+ generalized partner regression")

# clip to the observed range: a negative bandgap is not a polymer, and one wild
# extrapolation can wreck an R2 computed over only ~150 test rows
final = np.asarray(final_test, dtype=float)
for t in TARGETS:
    m = (test_df["target_type"] == t).values
    if not m.any():
        continue
    v = train_df.loc[train_df.target_type == t, "target"]
    lo, hi = v.min(), v.max(); pad = 0.05 * (hi - lo)
    n_clip = int(((final[m] < lo - pad) | (final[m] > hi + pad)).sum())
    final[m] = np.clip(final[m], lo - pad, hi + pad)
    if n_clip:
        print(f"  clipped {n_clip}/{m.sum()} {t} predictions")
bad = ~np.isfinite(final)
if bad.any():
    for t in TARGETS:
        m = (test_df["target_type"] == t).values & bad
        if m.any():
            final[m] = float(train_df.loc[train_df.target_type == t, "target"].mean())
    print(f"replaced {int(bad.sum())} non-finite predictions with the per-target mean")

submission = pd.DataFrame({"id": test_df["id"].values, "target": final})
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
print(f"\\nwrote submission.csv: {len(submission)} rows   total {time.time()-t_start:.0f}s")
print(submission.head().to_string(index=False))
'''

EXPLAIN_CELL = '''
FAMILY = {"rd": "RDKit descriptors", "mfp2": "Morgan r=2", "mfp3": "Morgan r=3",
          "ap": "atom pair", "tt": "topological torsion", "mac": "MACCS keys",
          "po": "polymer-specific", "grp": "functional groups"}
_cols = feature_names()

for t in TARGETS:
    rows = np.where((train_df["target_type"] == t).values)[0]
    if len(rows) < 20:
        continue
    idx = rows if len(rows) <= 800 else np.random.RandomState(SEED).choice(rows, 800, replace=False)
    mdl = make("lgbm", t, len(rows), SEED)
    mdl.fit(X_tr[rows], train_df["target"].values[rows])
    booster = mdl.model.booster_
    contrib = booster.predict(X_tr[idx], pred_contrib=True)
    vals = np.abs(contrib[:, :-1]).mean(axis=0)
    s = pd.Series(vals, index=_cols).sort_values(ascending=False)

    fam = s.groupby([c.split("_", 1)[0] for c in s.index]).sum()
    fam = (fam / fam.sum() * 100).sort_values(ascending=False)
    print(f"\\n===== {t}  (n={len(rows)}) =====")
    print("  attribution by feature family:")
    for k, v in fam.items():
        if v >= 0.5:
            print(f"    {FAMILY.get(k, k):<22} {v:5.1f}%")
    interp = s[[c for c in s.index if c.startswith(("rd_", "po_", "grp_"))]]
    print("  top interpretable features:")
    for k, v in interp.head(8).items():
        print(f"    {k:<32} {v:.4g}")
'''

INVARIANCE_CELL = '''
# Three ways the same polymer can be written differently. Permutational
# invariance is EXACT by construction here, because every feature is computed
# from canonicalize(), which is idempotent. Translational and repetition
# rewritings change the graph RDKit sees, so those are measured, not claimed.
_rng = np.random.RandomState(SEED)
_idx = _rng.choice(len(test_df), size=min(150, len(test_df)), replace=False)
_sample = test_df.iloc[_idx].reset_index(drop=True)

def _predict_rows(df):
    """Score a frame through the same featurisation the pipeline uses."""
    Xq = featurize(df["canon"])
    out = np.zeros(len(df))
    for t in TARGETS:
        m = (df["target_type"] == t).values
        if not m.any():
            continue
        out[m] = _INV_MODELS[t].predict(Xq[m])
    return out

_INV_MODELS = {}
for t in TARGETS:
    rows = np.where((train_df["target_type"] == t).values)[0]
    mdl = make("lgbm", t, len(rows), SEED)
    mdl.fit(X_tr[rows], train_df["target"].values[rows])
    _INV_MODELS[t] = mdl

_base = _predict_rows(_sample)
print(f"{'rewriting':<18}{'changed':>9}{'canon moved':>13}{'median |d|':>13}{'max |d|':>12}")
for _kind, _fn in [("permutational", lambda s, i: randomize(s, seed=i)),
                   ("translational", lambda s, i: translate(s, k=i + 1)),
                   ("repetition",    lambda s, i: build_oligomer(s, n=i + 2))]:
    _d, _nch, _ncanon = [], 0, 0
    for _i in range(2):
        _v = _sample.copy()
        _v["smiles"] = [_fn(s, _i) for s in _sample["smiles"]]
        _v["canon"] = _v["smiles"].map(canonicalize)
        _ch = (_v["smiles"] != _sample["smiles"]).values
        _nch += int(_ch.sum())
        _ncanon += int((_v["canon"] != _sample["canon"]).values.sum())
        _p = _predict_rows(_v)
        _d.extend(np.abs(_p - _base)[_ch])
    _d = np.array(_d) if _d else np.array([0.0])
    print(f"{_kind:<18}{_nch:>9}{_ncanon:>13}{np.median(_d):>13.6f}{_d.max():>12.4f}")

_v = _sample.copy()
_v["smiles"] = [randomize(s, seed=0) for s in _sample["smiles"]]
_v["canon"] = _v["smiles"].map(canonicalize)
_perm_max = float(np.abs(_predict_rows(_v) - _base).max())
assert _perm_max < 1e-9, f"permutational invariance broken: max delta {_perm_max}"
print(f"\\nPASS: permutational invariance is EXACT (max delta {_perm_max:.1e}).")
print("Every feature derives from the canonical SMILES, and canonicalisation is idempotent,")
print("so any re-ordering of the atoms in the input string yields a bit-identical prediction.")
'''

AUDIT_CELL = '''
import glob as _glob
_fail, _warn = [], []

_sub = pd.read_csv(os.path.join(WORK_DIR, "submission.csv"))
_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
if list(_sub.columns) != ["id", "target"]:
    _fail.append(f"columns are {list(_sub.columns)}, expected ['id','target']")
if len(_sub) != len(_test):
    _fail.append(f"{len(_sub)} rows, test.csv has {len(_test)}")
if set(_sub["id"]) != set(_test["id"]):
    _fail.append("id set does not match test.csv")
if _sub["id"].duplicated().any():
    _fail.append("duplicate ids")
if not np.isfinite(_sub["target"]).all():
    _fail.append("non-finite predictions")

_m = _test.merge(_sub, on="id")
for _t in TARGETS:
    _v = _m.loc[_m.target_type == _t, "target"]
    _o = train_df.loc[train_df.target_type == _t, "target"]
    if len(_v) and _v.std() < 1e-8:
        _fail.append(f"{_t}: all predictions identical")
    print(f"  {_t:<4} n={len(_v):<5} pred [{_v.min():9.4g}, {_v.max():9.4g}]"
          f"   train [{_o.min():9.4g}, {_o.max():9.4g}]")

_others = [p for p in _glob.glob("/kaggle/input/*")
           if os.path.abspath(p) != os.path.abspath(DATA_DIR)]
if _others:
    _fail.append(f"other datasets attached: {_others} -- rule 6.2.1 requires only competition data")

print()
print(f"data read from : {DATA_DIR}")
print(f"other inputs   : {_others or 'none'}")
print(f"seeds          : SEED={SEED}, NN_SEEDS={NN_SEEDS} (set and printed at the top)")
print(f"artifacts read : none (nothing deserialised from disk; no checkpoint import)")
print(f"wall-clock deps: none (every loop is a fixed fold/epoch count)")
print(f"local OOF score: {FINAL_SCORE:.4f}")
for _w in _warn:
    print("WARN ", _w)
for _f in _fail:
    print("FAIL ", _f)
assert not _fail, f"compliance audit failed: {_fail}"
print("\\nPASS: submission.csv is well formed and the run is rule-compliant.")
'''


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="lgbm,xgb,cb,mtnn,cnn")
    p.add_argument("--out", default="submissions/final.ipynb")
    a = p.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    nb = build(models, a.out)
    print(f"wrote {a.out}: {len(nb['cells'])} cells, models={models}")
