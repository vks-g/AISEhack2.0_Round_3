"""Exact competition metric: the UNWEIGHTED mean R^2 across the 7 target_types.

The target_type values in train.csv/test.csv are LOWERCASE. Getting this wrong
does not raise -- it silently matches zero rows and returns nan for every
target, so the assertion at import time is deliberate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Ordered by train-set size. Lowercase, exactly as they appear in the CSVs.
TARGETS: list[str] = ["tg", "egc", "egb", "eps", "nc", "ei", "eea"]

# Measured on train.csv -- used for sanity checks and clipping.
TARGET_RANGE: dict[str, tuple[float, float]] = {
    "tg":  (-109.82, 495.00),
    "egc": (0.0205, 9.8627),
    "egb": (0.5068, 10.1137),
    "eps": (2.6100, 9.0900),
    "nc":  (1.5596, 2.7581),
    "ei":  (4.0261, 9.8385),
    "eea": (0.3936, 5.1438),
}

TRAIN_COUNTS = {"tg": 4143, "egc": 2028, "egb": 337, "eps": 229, "nc": 229, "ei": 222, "eea": 221}
TEST_COUNTS = {"tg": 2763, "egc": 1352, "egb": 224, "eps": 153, "nc": 153, "ei": 148, "eea": 147}
N_TEST_ROWS = 4940  # NOT 4497 -- 4497 is the count of unique raw SMILES in test.csv


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - float(((y_true - y_pred) ** 2).sum()) / ss_tot


def competition_score(df: pd.DataFrame, y_col="target", pred_col="pred",
                      type_col="target_type") -> tuple[float, dict[str, float]]:
    """(mean_score, per_target). df is long format: one row per (polymer, target_type)."""
    seen = set(df[type_col].unique())
    unknown = seen - set(TARGETS)
    if unknown:
        raise ValueError(
            f"unrecognised target_type values {sorted(unknown)}. "
            f"Expected lowercase {TARGETS}. Check for a casing bug."
        )
    per: dict[str, float] = {}
    for t in TARGETS:
        m = (df[type_col] == t).values
        per[t] = r2(df.loc[m, y_col].values, df.loc[m, pred_col].values) if m.sum() else float("nan")
    vals = [v for v in per.values() if not np.isnan(v)]
    if not vals:
        raise ValueError("no target_type matched any row -- scored nothing.")
    return float(np.mean(vals)), per


def report(df, **kw) -> tuple[float, dict[str, float]]:
    score, per = competition_score(df, **kw)
    for t in TARGETS:
        n = int((df["target_type"] == t).sum())
        print(f"  {t:<4} n={n:<5} R2 = {per[t]:+.4f}")
    print(f"  {'MEAN':<4} {'':<7} = {score:+.4f}")
    return score, per
