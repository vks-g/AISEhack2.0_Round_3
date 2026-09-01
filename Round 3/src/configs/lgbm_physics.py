"""LightGBM + co-observed partner features + AFFINE-FITTED physics blending.

Three additions over `lgbm`, all aimed at the four ~220-row DFT properties that
make up 4/7 of the score:

  1. partner features -- the measured values of a polymer's OTHER properties,
     with `src.partners.drop_leaky` removing the row's own label.
  2. an affine-calibrated physics estimate. The raw identities are badly biased
     for eps (R2 0.336) and nc (0.171); fitting a*x+b on the training fold takes
     them to 0.855 and 0.837. See src/physics for the measured table.
  3. a blend weight tuned per target on an inner out-of-fold split of the
     TRAINING fold only, so the weight never sees validation labels.

Physics only touches rows whose partner values are actually present; everything
else falls through to the model prediction unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

from src import partners, physics
from src.configs.lgbm import params_for
from src.features import featurize
from src.metric import TARGETS

NAME = "lgbm_physics"
DESCRIPTION = "lgbm + co-observed partner features + affine physics blend"

INNER_FOLDS = 5


def _design(X, block, target_type):
    kept = partners.drop_leaky(block, target_type)
    return np.hstack([X, kept.values.astype(np.float32)])


def fit(train_df, seed: int = 42, targets=None):
    targets = list(targets) if targets else TARGETS
    X = featurize(train_df["canon"])
    (block,), _ = partners.build(train_df, [train_df])

    rels = physics.fit_relations(train_df)
    lookup = physics.wide_table(train_df)

    models, weights = {}, {}
    for t in targets:
        m = (train_df["target_type"] == t).values
        n = int(m.sum())
        if n < 5:
            continue
        Xt = _design(X[m], block.loc[m], t)
        y = train_df.loc[m, "target"].values
        models[t] = lgb.LGBMRegressor(**params_for(n, seed)).fit(Xt, y)

        # Inner OOF on the training fold -> a blend weight that never sees the
        # outer validation labels.
        if t in rels:
            inner = np.zeros(n)
            k = max(2, min(INNER_FOLDS, n))
            for a, b in KFold(k, shuffle=True, random_state=seed).split(Xt):
                inner[b] = lgb.LGBMRegressor(**params_for(len(a), seed)).fit(
                    Xt[a], y[a]).predict(Xt[b])
            sub = train_df.loc[m].reset_index(drop=True)
            est, _ = physics.partner_estimate(sub, {t: rels[t]}, lookup)
            w, _ = physics.tune_weight(y, inner, est)
            weights[t] = w

    fallback = {t: float(train_df.loc[train_df.target_type == t, "target"].mean())
                for t in TARGETS if (train_df.target_type == t).any()}
    return {"models": models, "weights": weights, "rels": rels,
            "lookup": lookup, "fallback": fallback, "train": train_df}


def predict(state, df):
    X = featurize(df["canon"])
    (block,), _ = partners.build(state["train"], [df])
    out = np.zeros(len(df))
    for t in df["target_type"].unique():
        m = (df["target_type"] == t).values
        if t not in state["models"]:
            out[m] = state["fallback"].get(t, 0.0)
            continue
        out[m] = state["models"][t].predict(_design(X[m], block.loc[m], t))
        w = state["weights"].get(t, 0.0)
        if w > 0 and t in state["rels"]:
            sub = df.loc[m].reset_index(drop=True)
            est, _ = physics.partner_estimate(sub, {t: state["rels"][t]}, state["lookup"])
            out[m] = physics.blend(out[m], est, w)
    return out
