"""Per-property gradient-boosting estimators: LightGBM, XGBoost, CatBoost.

Uniform factory so the CV driver can treat all three identically:

    from src.models import trees
    est = trees.make('lgbm', 'eps', n_rows=229, seed=42)
    est.fit(X_tr, y_tr).predict(X_va)

WHY CAPACITY IS A FUNCTION OF n_rows
------------------------------------
Per-target train sizes span a factor of 19 (tg 4139 ... eea 221) and every
target is worth the same 1/7 of the metric. One hyper-parameter set sized for
`tg` over-fits the four ~220-row DFT properties, which together are 4/7 of the
score. `_tier()` buckets n into small/mid/large and every builder switches on
it: fewer leaves, shallower trees, larger minimum leaf occupancy and stronger
L2 as n shrinks.

The lgbm branch is anchored on `src.configs.lgbm.params_for` so this module and
the reference config cannot drift apart; xgb and cb are capacity-matched
analogues (lgbm `num_leaves=31` ~ a depth-5 symmetric tree, `num_leaves=15` ~
depth 4).

COMPETITION-RULE NOTES
----------------------
* Trains from scratch on whatever it is handed. No checkpoints, no pickles, no
  downloads, no reading or writing files -- CatBoost's `allow_writing_files` is
  forced off so it cannot drop a `catboost_info/` directory.
* No wall-clock branching anywhere. Iteration counts are fixed constants.
* No early stopping inside `.fit()`: the driver owns the folds, and an
  early-stopping set carved out here would leak the fold's own validation rows.
* Deterministic for a fixed (seed, n_jobs). CatBoost and LightGBM both partition
  work by thread, so reproducing a score requires pinning `n_jobs` too --
  `DEFAULT_N_JOBS` is derived from `os.cpu_count()`, so pass it explicitly if
  local and Kaggle core counts differ and you need bit-identical output.
"""
from __future__ import annotations

import os

import numpy as np

from src.configs.lgbm import params_for as _lgbm_params_for

NAME = "trees"
ESTIMATORS = ["lgbm", "xgb", "cb"]

# Kaggle CPU sessions give 4 cores, this machine has 11. Derived, never hardcoded.
DEFAULT_N_JOBS = max(1, os.cpu_count() or 4)

SMALL_N = 400     # eps/nc/ei/eea (~220) and egb (337) land here
MID_N = 1500      # egc (2028) is 'large', nothing currently lands in 'mid'


def _tier(n_rows: int) -> str:
    if n_rows < SMALL_N:
        return "small"
    if n_rows < MID_N:
        return "mid"
    return "large"


def _jobs(n_jobs: int | None) -> int:
    return DEFAULT_N_JOBS if n_jobs is None else max(1, int(n_jobs))


# --------------------------------------------------------------------------- params

def lgbm_params(n_rows: int, seed: int, n_jobs: int | None = None) -> dict:
    """Reference `params_for` plus a middle tier and an explicit thread count."""
    p = dict(_lgbm_params_for(int(n_rows), int(seed)))
    if _tier(n_rows) == "mid":
        p.update(n_estimators=1100, num_leaves=23, max_depth=6, min_child_samples=8)
    p["n_jobs"] = _jobs(n_jobs)
    p["random_state"] = int(seed)
    return p


def xgb_params(n_rows: int, seed: int, n_jobs: int | None = None) -> dict:
    tier = _tier(n_rows)
    depth = {"small": 4, "mid": 5, "large": 6}[tier]
    return dict(
        n_estimators={"small": 900, "mid": 1100, "large": 1200}[tier],
        learning_rate=0.03,
        max_depth=depth,
        min_child_weight={"small": 5.0, "mid": 4.0, "large": 3.0}[tier],
        subsample=0.8,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda={"small": 3.0, "mid": 2.0, "large": 1.0}[tier],
        tree_method="hist",
        max_bin=256,
        objective="reg:squarederror",
        random_state=int(seed),
        n_jobs=_jobs(n_jobs),
        verbosity=0,
    )


def cb_params(n_rows: int, seed: int, n_jobs: int | None = None) -> dict:
    """CatBoost's symmetric trees cost ~2^depth leaves, so depth moves one step
    below the xgb depth-wise analogue. `rsm` is CatBoost's colsample; with 2978
    columns it is also the main runtime lever."""
    tier = _tier(n_rows)
    return dict(
        iterations={"small": 900, "mid": 1100, "large": 1200}[tier],
        learning_rate=0.04,
        depth={"small": 4, "mid": 5, "large": 6}[tier],
        l2_leaf_reg={"small": 6.0, "mid": 4.0, "large": 3.0}[tier],
        min_data_in_leaf={"small": 5, "mid": 8, "large": 10}[tier],
        rsm=0.3,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        border_count={"small": 64, "mid": 96, "large": 128}[tier],
        loss_function="RMSE",
        random_seed=int(seed),
        thread_count=_jobs(n_jobs),
        allow_writing_files=False,   # otherwise it drops catboost_info/ on disk
        # `verbose` and `logging_level` are mutually exclusive in catboost 1.2 --
        # setting both raises. 'Silent' is the one that also mutes the fit banner.
        logging_level="Silent",
    )


# --------------------------------------------------------------------------- wrapper

class _Boosted:
    """Uniform `.fit(X, y, smiles=None)` / `.predict(X, smiles=None)` shim.

    `smiles` is accepted and ignored: these are pure tabular learners, but the
    driver calls every estimator family through one signature and the SMILES
    models in sibling modules need it.
    """

    __slots__ = ("kind", "target_type", "n_rows", "seed", "params", "model", "_const")

    def __init__(self, kind, target_type, n_rows, seed, params, model):
        self.kind = kind
        self.target_type = target_type
        self.n_rows = int(n_rows)
        self.seed = int(seed)
        self.params = params
        self.model = model
        self._const = None

    def fit(self, X, y, smiles=None):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows, y has {y.shape[0]}")
        # Degenerate folds: a constant or near-empty target makes every booster
        # either warn loudly or fail outright. Predict the mean instead.
        if len(y) < 5 or float(np.ptp(y)) == 0.0:
            self._const = float(y.mean()) if len(y) else 0.0
            return self
        self._const = None
        self.model.fit(X, y)
        return self

    def predict(self, X, smiles=None):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        if self._const is not None:
            return np.full(X.shape[0], self._const, dtype=np.float64)
        return np.asarray(self.model.predict(X), dtype=np.float64).ravel()

    def feature_importance(self):
        """Gain-style importance, aligned with `src.features.feature_names()`.
        Returns None before `.fit()` or on a degenerate constant fit."""
        if self._const is not None:
            return None
        if self.kind == "lgbm":
            return np.asarray(self.model.booster_.feature_importance("gain"), dtype=np.float64)
        if self.kind == "xgb":
            return np.asarray(self.model.feature_importances_, dtype=np.float64)
        return np.asarray(self.model.get_feature_importance(), dtype=np.float64)

    def __repr__(self):
        return (f"<{self.kind} target={self.target_type} n={self.n_rows} "
                f"tier={_tier(self.n_rows)} seed={self.seed}>")


def make(kind: str, target_type: str, n_rows: int, seed: int, *, n_jobs: int | None = None):
    """Build one unfitted per-property booster.

    kind         one of ESTIMATORS
    target_type  the property this estimator serves; recorded, not branched on,
                 so capacity is decided by data size rather than by name
    n_rows       rows this estimator will be FIT on (i.e. the training fold, not
                 the whole target) -- that is what capacity must track
    seed         propagated to every source of randomness
    n_jobs       threads; defaults to DEFAULT_N_JOBS
    """
    kind = str(kind).lower()
    n_rows = int(n_rows)
    seed = int(seed)

    if kind == "lgbm":
        import lightgbm as lgb
        p = lgbm_params(n_rows, seed, n_jobs)
        model = lgb.LGBMRegressor(**p)
    elif kind == "xgb":
        from xgboost import XGBRegressor
        p = xgb_params(n_rows, seed, n_jobs)
        model = XGBRegressor(**p)
    elif kind == "cb":
        from catboost import CatBoostRegressor
        p = cb_params(n_rows, seed, n_jobs)
        model = CatBoostRegressor(**p)
    else:
        raise ValueError(f"unknown kind {kind!r}; expected one of {ESTIMATORS}")

    return _Boosted(kind, target_type, n_rows, seed, p, model)
