"""The host's baseline, reimplemented against this harness.

Ridge on RDKit descriptors, one model per property. Fast (seconds) -- use it as
the smoke test that the harness works end to end, not as a modelling candidate.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import feature_names, featurize
from src.metric import TARGETS

NAME = "ridge_baseline"
DESCRIPTION = "Ridge on RDKit descriptors only, per property (host baseline)"

_RD = np.array([i for i, c in enumerate(feature_names()) if c.startswith("rd_")])


def fit(train_df, seed: int = 42, targets=None):
    targets = list(targets) if targets else TARGETS
    X = featurize(train_df["canon"])[:, _RD]
    models = {}
    for t in targets:
        m = (train_df["target_type"] == t).values
        if m.sum() < 5:
            continue
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=seed))
        models[t] = pipe.fit(np.nan_to_num(X[m]), train_df.loc[m, "target"].values)
    fallback = {t: float(train_df.loc[train_df.target_type == t, "target"].mean())
                for t in TARGETS if (train_df.target_type == t).any()}
    return {"models": models, "fallback": fallback}


def predict(state, df):
    X = np.nan_to_num(featurize(df["canon"])[:, _RD])
    out = np.zeros(len(df))
    for t in df["target_type"].unique():
        m = (df["target_type"] == t).values
        if t in state["models"]:
            out[m] = state["models"][t].predict(X[m])
        else:
            out[m] = state["fallback"].get(t, 0.0)
    return out
