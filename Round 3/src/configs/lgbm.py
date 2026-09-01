"""Per-property LightGBM on the full feature set. The local reference model.

Hyperparameters scale with sample count: one setting sized for `tg` (4139 rows)
badly over-fits the four ~220-row properties, which are 4/7 of the score.
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb

from src.features import featurize
from src.metric import TARGETS

NAME = "lgbm"
DESCRIPTION = "per-property LightGBM, size-adaptive capacity, 2978 features"


def params_for(n: int, seed: int) -> dict:
    big = n > 1000
    return dict(
        objective="regression", metric="rmse", verbosity=-1,
        n_estimators=1200 if big else 900,
        learning_rate=0.03,
        num_leaves=31 if big else 15,
        max_depth=7 if big else 5,
        min_child_samples=10 if big else 5,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, n_jobs=-1,
    )


def fit(train_df, seed: int = 42, targets=None):
    targets = list(targets) if targets else TARGETS
    X = featurize(train_df["canon"])
    models = {}
    for t in targets:
        m = (train_df["target_type"] == t).values
        if m.sum() < 5:
            continue
        y = train_df.loc[m, "target"].values
        models[t] = lgb.LGBMRegressor(**params_for(int(m.sum()), seed)).fit(X[m], y)
    fallback = {t: float(train_df.loc[train_df.target_type == t, "target"].mean())
                for t in TARGETS if (train_df.target_type == t).any()}
    return {"models": models, "fallback": fallback}


def predict(state, df):
    X = featurize(df["canon"])
    out = np.zeros(len(df))
    for t in df["target_type"].unique():
        m = (df["target_type"] == t).values
        if t in state["models"]:
            out[m] = state["models"][t].predict(X[m])
        else:
            out[m] = state["fallback"].get(t, 0.0)
    return out
