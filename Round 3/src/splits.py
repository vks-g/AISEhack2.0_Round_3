"""Cross-validation splits.

MEASURED FACT that decides the design: within a single target_type there are
essentially no duplicate canonical polymers --

    tg   4143 rows / 4139 canonical   (4 duplicate rows)
    egc  2028 / 2028      egb 337 / 337     eps 229 / 229
    nc    229 /  229      ei  222 / 222     eea 221 / 221

So a plain KFold over the rows of ONE property is already leakage-safe, and it
is what the host's own baseline notebook does. Grouping only matters for a
MULTI-TASK model, where one polymer contributes several rows: 6150 polymers have
one property, 157 have two, 126 three, 100 four, 28 five, 4 six.

Use per_property_folds() for per-property models (the normal case) and
grouped_folds() for anything that trains on the full long table at once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

N_FOLDS_DEFAULT = 10
SMALL_N = 400          # properties below this get more folds -- less noisy OOF
N_FOLDS_SMALL = 15


def n_folds_for(n_rows: int, base: int = N_FOLDS_DEFAULT) -> int:
    """More folds for the ~220-row properties: bigger training fraction per fold
    and a less noisy OOF estimate, at negligible cost when n is this small."""
    return N_FOLDS_SMALL if n_rows < SMALL_N else base


def per_property_folds(df: pd.DataFrame, target_type: str, seed: int = 42,
                       n_folds: int | None = None):
    """Yield (train_idx, valid_idx) as positions into the rows of ONE property."""
    sub = df.index[df["target_type"] == target_type]
    n = len(sub)
    k = n_folds or n_folds_for(n)
    k = max(2, min(k, n))
    for tr, va in KFold(k, shuffle=True, random_state=seed).split(np.arange(n)):
        yield tr, va


def grouped_folds(df: pd.DataFrame, n_folds: int = N_FOLDS_DEFAULT, seed: int = 42):
    """Folds over the FULL long table, grouped so one polymer never straddles a
    split. Required for multi-task models, where a polymer contributes up to six
    rows. Yields positional indices into `df`.

    sklearn >= 1.6 gives GroupKFold a real `shuffle`/`random_state`; without it
    the split is a deterministic function of group sizes and the seed does
    nothing, which silently turns a multi-seed check into the same run twice.
    """
    groups = df["canon"].values
    try:
        gk = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    except TypeError:  # sklearn < 1.6
        gk = GroupKFold(n_splits=n_folds)
    for tr, va in gk.split(np.arange(len(df)), groups=groups):
        yield tr, va
