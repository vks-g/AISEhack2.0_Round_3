"""Cheap, structurally different learners for the stack.

The three gradient-boosting models agree with each other far more than they
agree with the truth -- adding a fourth booster buys almost nothing (xgb on top
of lgbm+cb was worth +0.0001). What the stack actually wants is a model that is
*wrong in different places*, and the cheapest sources of that are:

  knn   Tanimoto/Jaccard nearest neighbours over the Morgan bits. This is the
        chemist's prior -- similar substructure, similar property -- expressed
        directly rather than learned. It costs seconds and its errors are
        concentrated on structurally novel polymers, exactly where a tree
        extrapolates badly.
  et    Extra Trees. Randomised split thresholds give a very different bias from
        boosting's greedy ones.
  ridge Linear on the descriptor block only. Weak alone, but it extrapolates
        smoothly where trees produce flat plateaus.

Same interface as models/trees.py: make(kind, target_type, n_rows, seed).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NAME = "simple"
ESTIMATORS = ["knn", "et", "ridge"]

# Column layout of src/features.py, needed to slice out the fingerprint block.
_N_DESC = 217
_MFP2 = (_N_DESC, _N_DESC + 1024)


class _Wrapped:
    __slots__ = ("model", "cols", "kind")

    def __init__(self, model, cols, kind):
        self.model, self.cols, self.kind = model, cols, kind

    def _slice(self, X):
        return X if self.cols is None else X[:, self.cols[0]:self.cols[1]]

    def fit(self, X, y, smiles=None):
        Xs = self._slice(np.asarray(X, dtype=np.float32))
        self.model.fit(np.nan_to_num(Xs), np.asarray(y, dtype=float))
        return self

    def predict(self, X, smiles=None):
        Xs = self._slice(np.asarray(X, dtype=np.float32))
        return np.asarray(self.model.predict(np.nan_to_num(Xs)), dtype=float)


def make(kind: str, target_type: str, n_rows: int, seed: int, n_jobs: int | None = None):
    n_jobs = -1 if n_jobs is None else n_jobs
    if kind == "knn":
        # Jaccard on the binary Morgan block IS Tanimoto similarity. Neighbour
        # count scales with n so the ~220-row targets are not over-smoothed.
        k = int(np.clip(round(n_rows ** 0.4), 3, 25))
        return _Wrapped(
            KNeighborsRegressor(n_neighbors=k, metric="jaccard",
                                weights="distance", n_jobs=n_jobs),
            _MFP2, kind)
    if kind == "et":
        return _Wrapped(
            ExtraTreesRegressor(
                n_estimators=600,
                max_features=0.3,
                min_samples_leaf=1 if n_rows > 1000 else 2,
                bootstrap=False, random_state=seed, n_jobs=n_jobs),
            None, kind)
    if kind == "ridge":
        return _Wrapped(
            make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=seed)),
            (0, _N_DESC), kind)
    raise ValueError(f"unknown kind {kind!r}")
