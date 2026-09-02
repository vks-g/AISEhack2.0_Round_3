"""Out-of-fold engine: base models -> cross-fitted stack -> two-stage physics.

The whole pipeline in one place, so the number this prints is the number the
notebook reproduces.

Three things here are deliberately cross-fitted rather than fitted on the full
OOF. Fitting a stacker or a blend weight on the same out-of-fold predictions you
then score is the classic way to get an OOF number that does not survive contact
with the leaderboard, and it is the leading explanation for the team's
0.903 OOF -> 0.883 public LB gap:

  1. the Ridge stacker            cross-fitted over `fold_id`
  2. the physics blend weight     cross-fitted over `fold_id`
  3. the physics calibration a,b  fitted on training-fold rows only

FOLDS. One global fold assignment, grouped on canonical SMILES, shared by every
base model. Grouping is only strictly required for the multi-task models (a
polymer contributes up to six rows there), but sharing one assignment is what
makes stacking valid: every base model's OOF for a given row comes from a model
that saw exactly the same training rows.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src import physics
from src.data import load_test, load_train
from src.features import featurize
from src.metric import TARGETS, competition_score, r2
from src.paths import REPO, cache_dir
from src.splits import grouped_folds

N_FOLDS = 10
SEED = 42
AUG_MAX_N = 400        # only targets smaller than this get repeat-unit augmentation
STACK_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


# --------------------------------------------------------------------------- #
# folds                                                                        #
# --------------------------------------------------------------------------- #

def folds_for_target(n_rows: int, base: int = N_FOLDS) -> int:
    """Fold count scaled to sample size, in the direction runtime wants.

    tg (4139 rows) and egc (2028) sit at R2 ~0.91 and are the two expensive
    targets: extra folds buy almost nothing there and dominate wall time. The
    four ~220-row targets are noisy and cheap, so they get more folds -- a bigger
    training fraction per fold and a less noisy OOF estimate, for free.
    """
    # ONE fold count everywhere. With fold-averaged test predictions each member
    # trains on (K-1)/K of the data, so a small K is no longer just a cheaper
    # estimate -- it is a weaker deployed model. The teammate notebook that
    # transfers best (-0.020 CV-to-LB vs our -0.041) uses a single KFold(10)
    # partition shared by every target and every model.
    return base


def build_fold_id(train_df: pd.DataFrame, n_folds: int = N_FOLDS,
                  seed: int = SEED, scheme: str = "per_property") -> np.ndarray:
    """Fold assignment for every training row.

    scheme="per_property" (the default, and the correct one here): KFold within
    each target_type independently. Leakage-safe, because within a target_type
    there are essentially no duplicate canonical polymers.

    scheme="grouped": one polymer never straddles a fold.

    WHY per_property AND NOT grouped, measured:

        partner availability for validation rows, grouped folds ... 0%
        partner availability for test rows, actual test set ....... 88-99%

    A polymer held out entirely loses the other properties measured on it, so
    grouped CV evaluates a model that has been denied a signal present for
    almost every real test row in the DFT block. It is not "conservative", it
    measures a different problem. Under per_property folds a held-out ei row
    keeps its polymer's egc and eea rows in training -- exactly the situation at
    test time, where 98% of ei test polymers appear in train under some property.

    The same argument covers the multi-task models: at test time the network has
    already seen a test polymer's structure whenever that polymer carries some
    other measured property, so per_property folds mirror inference there too.
    """
    fold_id = np.full(len(train_df), -1, dtype=int)
    if scheme == "grouped":
        for k, (_, va) in enumerate(grouped_folds(train_df, n_folds=n_folds, seed=seed)):
            fold_id[va] = k
    elif scheme == "per_property":
        from sklearn.model_selection import KFold
        for t in TARGETS:
            rows = np.where((train_df["target_type"] == t).values)[0]
            if len(rows) == 0:
                continue
            k_eff = max(2, min(folds_for_target(len(rows), n_folds), len(rows)))
            for k, (_, va) in enumerate(
                    KFold(k_eff, shuffle=True, random_state=seed).split(rows)):
                fold_id[rows[va]] = k
    else:
        raise ValueError(f"unknown fold scheme {scheme!r}")
    assert (fold_id >= 0).all(), "every row must land in exactly one fold"
    return fold_id


# --------------------------------------------------------------------------- #
# base models                                                                  #
# --------------------------------------------------------------------------- #


def physics_feature_cols(target, block, raw_only=False):
    """The fitted physics estimate as an explicit FEATURE, plus a validity flag.

    Partner features alone are not enough. A tree can see `true_egc` and
    `true_eea` but cannot cheaply represent `ei = egc + eea` -- axis-aligned
    splits approximate a sum badly. Handing the model the computed relation as
    one column lets it learn *when to trust it* per row, which is strictly more
    expressive than a single global blend weight applied afterwards.

    Returns (raw_estimate, all_partners_measured) or (None, None) when the
    target has no relation. The caller calibrates a*x+b on training rows.
    """
    rel = physics.RELATIONS.get(target)
    if rel is None:
        return None, None
    srcs, expr = rel
    cols = [f"true_{s}" for s in srcs]
    flags = [f"has_{s}" for s in srcs]
    if not all(c in block.columns for c in cols):
        return None, None
    raw = expr(block[cols].values.astype(float))
    have = block[flags].values.astype(bool).all(axis=1) if all(
        f in block.columns for f in flags) else np.ones(len(block), bool)
    return np.asarray(raw, dtype=float), have


def per_property_oof(kind_maker, kind: str, train_df, X_tr, test_df, X_te,
                     fold_id, all_canon, X_all, seed=SEED, n_jobs=None,
                     use_partners=True, use_physics_feature=True,
                     use_partner_ridge=True, use_augment=True, aug_max_n=AUG_MAX_N):
    """OOF + test + whole-universe predictions for one per-property estimator.

    Test and universe predictions are the AVERAGE OF THE K FOLD MODELS, not a
    separate model refit on 100% of the rows. That matters for two reasons:

      * calibration. The stack's Ridge coefficients and the physics blend weights
        are fitted on OOF columns produced by (K-1)/K-trained models. Feeding
        them a column from a 100%-trained model at inference means the thing they
        were calibrated on and the thing they consume are different objects.
      * variance. Bagging K models is free variance reduction we were giving up
        on three of our four base models -- only the multi-task NN averaged folds.

    The teammate notebook that transfers best (-0.020 CV-to-LB against our -0.041)
    averages ~80 fold models per test row and refits nothing.

    Partner features are built per fold from TRAINING rows only, and `drop_leaky`
    removes the column that would be the row's own label. The test and universe
    partner blocks come from the full training set, which is exactly what is
    available at inference.
    """
    from src import partners as P
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold as _KF
    _ALPHAS = [0.1, 1.0, 10.0, 100.0]

    oof = np.full(len(train_df), np.nan)
    test_pred = np.full(len(test_df), np.nan)
    universe = {t: np.full(len(all_canon), np.nan) for t in TARGETS}
    uni_df = pd.DataFrame({"canon": all_canon})
    y_all = train_df["target"].values

    for t in TARGETS:
        rows = np.where((train_df["target_type"] == t).values)[0]
        if len(rows) < 10:
            continue
        f = fold_id[rows]
        te = np.where((test_df["target_type"] == t).values)[0]

        # REPEAT-UNIT AUGMENTATION. `*CC*` and `*CCCC*` are the same polymer and
        # carry the same property value, so the dimer of a training row is a
        # genuinely new view of a known label -- free data exactly where we are
        # starved. Only the small targets get it: they are 5/7 of the metric on
        # under 5% of the rows, and augmenting tg/egc would double the most
        # expensive training for nothing. Augmented rows enter TRAINING FOLDS
        # ONLY, so no fold is ever scored on a row derived from its own
        # validation data. Measured on eps with lgbm: 0.7882 -> 0.8230.
        X_aug = None
        if use_augment and len(rows) < aug_max_n:
            from src.smiles_utils import build_oligomer, canonicalize
            dimers = [canonicalize(build_oligomer(c, 2))
                      for c in train_df["canon"].values[rows]]
            X_aug = featurize(dimers)
        pos_of_row = {r: i for i, r in enumerate(rows)}

        if use_partners:
            (bte, buni), _ = P.build(train_df, [test_df, uni_df])
            kte = P.drop_leaky(bte.iloc[te], t)
            kuni = P.drop_leaky(buni, t)
            Zte = np.nan_to_num(kte.values.astype(np.float64))
            Zuni = np.nan_to_num(kuni.values.astype(np.float64))

        fold_test, fold_uni = [], []
        for k in np.unique(f):
            tr = rows[f != k]
            va = rows[f == k]
            if len(tr) < 5:
                continue

            extra_a = []          # engineered columns appended to the design
            if use_partners:
                keep = np.ones(len(train_df), dtype=bool)
                keep[va] = False
                (btr, bva), _ = P.build(train_df[keep],
                                        [train_df.iloc[tr], train_df.iloc[va]])
                ktr, kva = P.drop_leaky(btr, t), P.drop_leaky(bva, t)
                extra_a.append(ktr.values)
                Xb = np.hstack([X_tr[va], kva.values])
                Xt = np.hstack([X_te[te], kte.values])
                Xu = np.hstack([X_all, kuni.values])

                if use_partner_ridge:
                    Ztr = np.nan_to_num(ktr.values.astype(np.float64))
                    Zva = np.nan_to_num(kva.values.astype(np.float64))
                    if len(Ztr) >= 30 and Ztr.shape[1]:
                        mdl = RidgeCV(alphas=_ALPHAS).fit(Ztr, y_all[tr])
                        inner = np.zeros(len(tr))
                        for ia, ib in _KF(5, shuffle=True, random_state=0).split(Ztr):
                            inner[ib] = RidgeCV(alphas=_ALPHAS).fit(
                                Ztr[ia], y_all[tr][ia]).predict(Ztr[ib])
                        extra_a.append(inner.reshape(-1, 1))
                        Xb = np.hstack([Xb, mdl.predict(Zva).reshape(-1, 1)])
                        Xt = np.hstack([Xt, mdl.predict(Zte).reshape(-1, 1)])
                        Xu = np.hstack([Xu, mdl.predict(Zuni).reshape(-1, 1)])

                if use_physics_feature:
                    ra, ha_ = physics_feature_cols(t, ktr)
                    rb, hb = physics_feature_cols(t, kva)
                    rt, ht = physics_feature_cols(t, kte)
                    ru, hu = physics_feature_cols(t, kuni)
                    if ra is not None:
                        ok = np.isfinite(ra) & ha_
                        if ok.sum() >= 15 and np.std(ra[ok]) > 1e-12:
                            a_, b_ = np.polyfit(ra[ok], y_all[tr][ok], 1)
                            extra_a.append(np.column_stack([a_ * ra + b_,
                                                            ha_.astype(np.float32)]))
                            Xb = np.hstack([Xb, np.column_stack([a_ * rb + b_, hb.astype(np.float32)])])
                            Xt = np.hstack([Xt, np.column_stack([a_ * rt + b_, ht.astype(np.float32)])])
                            Xu = np.hstack([Xu, np.column_stack([a_ * ru + b_, hu.astype(np.float32)])])
                Xa = np.hstack([X_tr[tr]] + extra_a)
            else:
                Xa, Xb = X_tr[tr], X_tr[va]
                Xt, Xu = X_te[te], X_all

            ya = y_all[tr]
            if X_aug is not None:
                # same molecule -> identical engineered columns, dimer structure
                tr_pos = [pos_of_row[r] for r in tr]
                Xa = np.vstack([Xa, np.hstack([X_aug[tr_pos]] + extra_a)])
                ya = np.concatenate([ya, y_all[tr]])

            m = kind_maker(kind, t, len(ya), seed)
            m.fit(Xa, ya)
            oof[va] = m.predict(Xb)
            fold_test.append(m.predict(Xt))
            fold_uni.append(m.predict(Xu))
            del Xa, Xb, Xt, Xu

        if fold_test and len(te):
            test_pred[te] = np.mean(np.column_stack(fold_test), axis=1)
        if fold_uni:
            universe[t] = np.mean(np.column_stack(fold_uni), axis=1)
        del fold_test, fold_uni
        gc.collect()

    return oof, test_pred, universe


# --------------------------------------------------------------------------- #
# stacking                                                                     #
# --------------------------------------------------------------------------- #

def _fit_ridge(Z, y, alphas=STACK_ALPHAS):
    """Pick alpha by a small internal CV on the meta-features."""
    from sklearn.model_selection import KFold
    best, best_a = -np.inf, alphas[0]
    if len(y) >= 20:
        for a in alphas:
            p = np.zeros(len(y))
            for tr, va in KFold(5, shuffle=True, random_state=0).split(Z):
                p[va] = Ridge(alpha=a).fit(Z[tr], y[tr]).predict(Z[va])
            s = r2(y, p)
            if s > best:
                best, best_a = s, a
    return Ridge(alpha=best_a).fit(Z, y), best_a


def stack(train_df, base_oof: dict, base_test: dict, test_df, fold_id,
          mode: str = "auto"):
    """Combine the base models, per target.

    MEASURED: with three correlated gradient-boosting models, a plain mean beats
    a cross-fitted Ridge meta-learner (0.8961 vs 0.8950 after physics), and Ridge
    over a *single* base model is worse than that model alone (0.8622 vs 0.8645).
    A meta-learner has to earn its variance, and on a 220-row target with inputs
    that correlate at ~0.99 it does not.

    mode="auto" therefore starts from the mean and only tries the cross-fitted
    Ridge once there are at least three base models, keeping it per target only
    where it actually beats the mean out-of-fold. Measured on lgbm+xgb+cb it
    selects ridge for tg, egc and eea and the mean for the rest, worth +0.0004.
    """
    names = sorted(base_oof)
    stacked_oof = np.full(len(train_df), np.nan)
    stacked_test = np.full(len(test_df), np.nan)
    chosen = {}

    use_ridge_allowed = (mode == "ridge") or (mode == "auto" and len(names) >= 3)

    for t in TARGETS:
        rows = np.where((train_df["target_type"] == t).values)[0]
        if len(rows) == 0:
            continue
        te = np.where((test_df["target_type"] == t).values)[0]
        Z = np.nan_to_num(np.column_stack([base_oof[n][rows] for n in names]))
        Zt = np.nan_to_num(np.column_stack([base_test[n][te] for n in names])) \
            if len(te) else np.zeros((0, len(names)))
        y = train_df["target"].values[rows]
        f = fold_id[rows]

        mean_oof = Z.mean(axis=1)
        pick = "mean"

        if use_ridge_allowed and len(rows) >= 60:
            Zx = np.column_stack([Z, Z.std(axis=1)])
            ridge_oof = np.full(len(rows), np.nan)
            for k in np.unique(f):
                tr_m, va_m = f != k, f == k
                if tr_m.sum() < 20 or va_m.sum() == 0:
                    continue
                mdl, _ = _fit_ridge(Zx[tr_m], y[tr_m])
                ridge_oof[va_m] = mdl.predict(Zx[va_m])
            if np.isfinite(ridge_oof).all() and r2(y, ridge_oof) > r2(y, mean_oof):
                pick = "ridge"
                stacked_oof[rows] = ridge_oof
                if len(te):
                    full, _ = _fit_ridge(Zx, y)
                    Ztx = np.column_stack([Zt, Zt.std(axis=1)])
                    stacked_test[te] = full.predict(Ztx)

        if pick == "mean":
            stacked_oof[rows] = mean_oof
            if len(te):
                stacked_test[te] = Zt.mean(axis=1)
        chosen[t] = pick

    return stacked_oof, stacked_test, chosen


# --------------------------------------------------------------------------- #
# two-stage physics                                                            #
# --------------------------------------------------------------------------- #

def partner_frame(train_df, all_canon, universe_by_target):
    """canon -> partner value per property, TRUE where measured else predicted.

    Stage 1 of the blend uses the measured value; stage 2 falls back to the
    model's prediction, which is what extends coverage from ~35-62% of rows to
    100%. `universe_by_target[t]` must be a prediction for every canonical
    polymer from a model that never saw that polymer's label for t -- polymers
    without a t label were never in that model's training set, so a full fit is
    legitimately out-of-sample for them.
    """
    idx = {c: i for i, c in enumerate(all_canon)}
    from src.partners import label_pool
    true_w = label_pool(train_df).pivot_table(
        index="canon", columns="target_type", values="target", aggfunc="mean")
    out = pd.DataFrame(index=all_canon)
    is_true = pd.DataFrame(index=all_canon)
    for t in TARGETS:
        pred = np.asarray(universe_by_target[t], dtype=float)
        col = pd.Series(pred, index=all_canon)
        if t in true_w.columns:
            tv = true_w[t].reindex(all_canon)
            have = tv.notna().values
            col = col.where(~have, tv)
            is_true[t] = have
        else:
            is_true[t] = False
        out[t] = col.values
    return out, is_true


def refine_universe(partners, is_true, train_df, n_rounds=3, damp=0.5,
                    verbose=False):
    """Belief propagation over the property graph, on the predicted entries only.

    The staged blend showed that even rows with NO measured partner take a large
    physics weight (0.63-0.77), i.e. the relation applied to *predicted* partners
    beats the direct model. If a predicted partner is that useful, it is worth
    improving before it is used: the seven properties form a small constraint
    graph (ei = egc+eea, eea = ei-egc, egb ~ egc, eps ~ nc^2, nc ~ sqrt(eps)) and
    one pass of the model's predictions does not satisfy it.

    Each round re-estimates every predicted cell from its neighbours and damps
    toward it. MEASURED cells are never overwritten -- they are the boundary
    conditions the propagation is anchored on. eps and nc are mutual inverses, so
    damping is required; undamped updates oscillate.
    """
    P = partners.copy()
    for rnd in range(n_rounds):
        updates = {}
        for t, (srcs, expr) in physics.RELATIONS.items():
            if t not in P.columns or any(c not in P.columns for c in srcs):
                continue
            # calibrate on train polymers where t IS measured
            meas = train_df[train_df["target_type"] == t]
            idx = meas["canon"].values
            src = P.reindex(idx)[srcs].values
            ok = np.isfinite(src).all(axis=1)
            if ok.sum() < 25:
                continue
            x = expr(src[ok])
            y = meas["target"].values[ok]
            if not np.isfinite(x).all() or np.std(x) < 1e-12:
                continue
            a, b = np.polyfit(x, y, 1)
            est = a * expr(P[srcs].values) + b
            cur = P[t].values.astype(float)
            upd = np.where(np.isfinite(est), (1 - damp) * cur + damp * est, cur)
            updates[t] = np.where(is_true[t].values, cur, upd)
        for t, v in updates.items():
            P[t] = v
        if verbose:
            print(f"  refine round {rnd+1}: updated {list(updates)}")
    return P


SHRINK = 0.75          # applied to every fitted blend weight before it is used


def ionic_term(train_df, X_tr, all_canon, X_all, fold_id, seed=SEED, verbose=True):
    """Predict the IONIC part of the dielectric constant from structure.

    DFPT computes the static dielectric constant as two separate contributions:

        eps  =  eps_electronic  +  eps_ionic
                = n^2 (Maxwell)   + lattice polarisation

    so `eps - nc^2` is not an empirical residual, it is the ionic term, and it is
    a physical quantity with its own structure-property relationship. VERIFIED on
    this data: `eps - nc^2` is POSITIVE for all 134 co-observed polymers (min
    +0.024). A generic residual would cross zero; a physical ionic contribution
    cannot be negative.

    The affine fit this replaces had already converged on a = 1.040, b = 0.615 --
    a is 1.0 and b is the mean ionic term (0.767). It was rediscovering the
    decomposition blindly, with the ionic part forced to a single constant.
    Naming it correctly makes it PREDICTABLE instead: the ionic term itself
    models at R2 0.708, and on the covered eps rows

        direct eps model                 0.7129
        nc^2 + constant ionic            0.8514      <- what a global affine gives
        nc^2 + structure-predicted ionic 0.9573

    Returns (per_fold, full): `per_fold[k]` is an array over `all_canon` from a
    model that never saw fold k's eps rows; `full` is fitted on all of them and
    is what the test side uses.
    """
    from src.models.trees import make

    w = train_df.pivot_table(index="canon", columns="target_type",
                             values="target", aggfunc="mean")
    if "eps" not in w.columns or "nc" not in w.columns:
        return {}, None
    d = w.dropna(subset=["eps", "nc"])
    if len(d) < 40:
        return {}, None
    ion = (d["eps"] - d["nc"] ** 2).values
    canon_ion = d.index.to_numpy()

    pos = {c: i for i, c in enumerate(all_canon)}
    keep = [i for i, c in enumerate(canon_ion) if c in pos]
    canon_ion, ion = canon_ion[keep], ion[keep]
    Xi = X_all[[pos[c] for c in canon_ion]]

    # each ionic training polymer inherits the fold of its own eps row, so a
    # fold's model never sees the eps values it will be used to reconstruct
    eps_rows = np.where((train_df["target_type"] == "eps").values)[0]
    fold_of = {train_df["canon"].values[r]: fold_id[r] for r in eps_rows}
    f_ion = np.array([fold_of.get(c, -1) for c in canon_ion])

    per_fold = {}
    for k in np.unique(fold_id):
        tr_m = f_ion != k
        if tr_m.sum() < 30:
            continue
        m = make("lgbm", "eps", int(tr_m.sum()), seed)
        m.fit(Xi[tr_m], ion[tr_m])
        per_fold[int(k)] = m.predict(X_all)
    full_m = make("lgbm", "eps", len(ion), seed)
    full_m.fit(Xi, ion)
    full = full_m.predict(X_all)
    if verbose:
        print(f"ionic term: fitted on {len(ion)} co-observed polymers, "
              f"mean {ion.mean():.3f} (all positive: {bool((ion > 0).all())})")
    return per_fold, full


def apply_physics(train_df, oof, test_df, test_pred, fold_id, partners, is_true,
                  oof_by_target=None, n_rounds=0, damp=0.5, shrink=SHRINK,
                  ionic=None, verbose=True):
    """Two-pass, cross-fitted, shrunk physics blend.

    TWO PASSES, not per-level staging. Rows split binary: every relation source
    measured, or not. Our previous version bucketed rows by HOW MANY sources were
    measured and fitted a separate affine and a separate weight per (fold, level)
    on as few as 15 rows. Measured, that bought nothing -- a single weight per
    target scored 0.9006 against 0.9005 for the staged version -- while fitting
    ~3x the free parameters on ~220-row targets and moving test predictions ~30%
    further. Same score, more ways to be wrong.

    SHRINKAGE. An argmax over a 21-point grid chosen on 55-135 rows is biased high
    by construction; cross-fitting averages that noise but does not remove the
    bias. Cutting every weight to 75% of its fitted value is free locally
    (0.9014 -> 0.9020 measured) and cuts test-side perturbation by 25-33%. The
    constant is taken from the teammate notebook that transfers best.

    The blend weight applied to test is the cross-fitted one, never re-tuned here
    against an estimate fitted to the same rows -- that bug degraded only the test
    side and was invisible in CV.
    """
    oof = np.asarray(oof, dtype=float)
    oof_out = oof.copy()
    test_out = np.asarray(test_pred, dtype=float).copy()
    info = {}
    canon_pos = {c: i for i, c in enumerate(partners.index)}
    ion_folds, ion_full = ionic if ionic else ({}, None)

    def _decomposed(target, canon_arr, src_vals, fold_k):
        """eps = n^2 + ionic, or nc = sqrt(eps - ionic). Returns None if the
        ionic model is unavailable, so the caller falls back to the affine fit."""
        vec = ion_folds.get(fold_k) if fold_k is not None else ion_full
        if vec is None:
            vec = ion_full
        if vec is None:
            return None
        ii = [canon_pos.get(c, -1) for c in canon_arr]
        if any(i < 0 for i in ii):
            return None
        ionv = np.asarray(vec)[ii]
        if target == "eps":
            return src_vals[:, 0] ** 2 + ionv
        if target == "nc":
            return np.sqrt(np.clip(src_vals[:, 0] - ionv, 1e-6, None))
        return None

    for t, (srcs, expr) in physics.RELATIONS.items():
        rows = np.where((train_df["target_type"] == t).values)[0]
        if len(rows) < 30:
            continue
        y = train_df["target"].values[rows]
        canon = train_df["canon"].values[rows]
        f = fold_id[rows]

        covered = is_true.reindex(canon)[srcs].all(axis=1).values
        blended = oof_out[rows].copy()
        weights = {}

        for k in np.unique(f):
            va = f == k
            if va.sum() == 0:
                continue
            Pk, Ik = partners, is_true
            pos = [(canon_pos[c], v) for c, v in zip(canon[va], oof[rows[va]])
                   if c in canon_pos]
            if pos:
                Pk = partners.copy(); Ik = is_true.copy()
                col = Pk[t].values.astype(float)
                col[[i for i, _ in pos]] = [v for _, v in pos]
                Pk[t] = col
                flag = Ik[t].values.copy(); flag[[i for i, _ in pos]] = False
                Ik[t] = flag
            cov_k = Ik.reindex(canon)[srcs].all(axis=1).values

            for name, mask in (("covered", cov_k), ("uncovered", ~cov_k)):
                tr_m = mask & (~va)
                va_m = mask & va
                if tr_m.sum() < 25 or va_m.sum() == 0:
                    continue
                rel = _fit_relation(train_df, rows[tr_m], t, srcs, expr, Pk)
                if rel is None:
                    continue
                e_tr = rel.apply(Pk.reindex(canon[tr_m])[srcs].values)
                e_va = rel.apply(Pk.reindex(canon[va_m])[srcs].values)
                if t in ("eps", "nc"):
                    _dt = _decomposed(t, canon[tr_m],
                                      Pk.reindex(canon[tr_m])[srcs].values, int(k))
                    _dv = _decomposed(t, canon[va_m],
                                      Pk.reindex(canon[va_m])[srcs].values, int(k))
                    if _dt is not None and _dv is not None:
                        e_tr, e_va = _dt, _dv
                w, _ = physics.tune_weight(y[tr_m], oof[rows[tr_m]], e_tr)
                w *= shrink
                blended[va_m] = physics.blend(oof[rows[va_m]], e_va, w)
                weights.setdefault(name, []).append(w)

        gain = r2(y, blended) - r2(y, oof[rows])
        info[t] = {"r2_before": r2(y, oof[rows]), "r2_after": r2(y, blended),
                   "gain": gain,
                   "n_covered": int(covered.sum()),
                   "n_uncovered": int((~covered).sum()),
                   "w": {k: round(float(np.mean(v)), 3) for k, v in weights.items()}}
        if gain > 0:
            oof_out[rows] = blended

        te = np.where((test_df["target_type"] == t).values)[0]
        if len(te) and gain > 0:
            tcanon = test_df["canon"].values[te]
            t_cov = is_true.reindex(tcanon)[srcs].all(axis=1).values
            for name, mask_tr, mask_te in (("covered", covered, t_cov),
                                           ("uncovered", ~covered, ~t_cov)):
                wl = weights.get(name)
                if not wl or mask_tr.sum() < 25 or mask_te.sum() == 0:
                    continue
                rel = _fit_relation(train_df, rows[mask_tr], t, srcs, expr, partners)
                if rel is None:
                    continue
                e_te = rel.apply(partners.reindex(tcanon[mask_te])[srcs].values)
                if t in ("eps", "nc"):
                    _dte = _decomposed(t, tcanon[mask_te],
                                       partners.reindex(tcanon[mask_te])[srcs].values, None)
                    if _dte is not None:
                        e_te = _dte
                test_out[te[mask_te]] = physics.blend(
                    test_out[te[mask_te]], e_te, float(np.mean(wl)))

    if verbose and info:
        print(f"\n{'target':<6}{'before':>9}{'after':>9}{'gain':>9}"
              f"{'covered':>10}{'uncov':>8}   weights (already shrunk)")
        for t, d in info.items():
            print(f"{t:<6}{d['r2_before']:>9.4f}{d['r2_after']:>9.4f}{d['gain']:>+9.4f}"
                  f"{d['n_covered']:>10}{d['n_uncovered']:>8}   {d['w']}")
    return oof_out, test_out, info


def _fit_relation(train_df, row_idx, target, srcs, expr, partners):
    """Calibrate a*x+b on the given TRAINING rows only."""
    canon = train_df["canon"].values[row_idx]
    y = train_df["target"].values[row_idx]
    src = partners.reindex(canon)[srcs].values
    ok = np.isfinite(src).all(axis=1) & np.isfinite(y)
    if ok.sum() < 15:
        return None
    x = expr(src[ok])
    if not np.isfinite(x).all() or np.std(x) < 1e-12:
        return None
    a, b = np.polyfit(x, y[ok], 1)
    return physics.Relation(target, list(srcs), expr, float(a), float(b), int(ok.sum()))


def partner_regression(train_df, pred, test_df, test_pred, fold_id, partners,
                       is_true, shrink=SHRINK, verbose=True):
    """Generalise the hand-written relations to a learned partner combination.

    `physics.RELATIONS` encodes the five relations a chemist can write down:
    ei = egc+eea, egb ~ egc, eps ~ nc^2 and so on. But a polymer's other measured
    properties carry more signal than those five expressions extract -- the whole
    DFT block is one electronic-structure description, so `eps` is informative
    about `nc` *and* about `ei`, not only through the Maxwell relation.

    So per target: ridge-regress the target on ALL other properties' values plus
    a measured/predicted flag for each, cross-fitted over the folds, and blend the
    result in with a cross-fitted weight. This subsumes the hand relations rather
    than replacing them -- it runs after them, on their output.

    MEASURED, on the four-model stack: +0.0022 mean, concentrated exactly where
    the score gap is -- eps +0.0077, nc +0.0067, egc +0.0012, everything else
    flat. The estimate itself reaches R2 0.84 on eps and 0.90 on nc from partner
    values alone, with no molecular features at all.

    No cycle, so no mask is needed: the target's own column is excluded from the
    design matrix, and the predicted entries in the other columns come from models
    that never saw this row's label for this target.
    """
    from sklearn.linear_model import RidgeCV

    out = np.asarray(pred, dtype=float).copy()
    test_out = np.asarray(test_pred, dtype=float).copy()
    info = {}
    alphas = [0.1, 1.0, 10.0, 100.0]

    for t in TARGETS:
        rows = np.where((train_df["target_type"] == t).values)[0]
        if len(rows) < 60:
            continue
        srcs = [c for c in TARGETS if c != t]
        canon = train_df["canon"].values[rows]
        y = train_df["target"].values[rows]
        f = fold_id[rows]

        def design(idx_canon):
            return np.nan_to_num(np.column_stack([
                partners.reindex(idx_canon)[srcs].values,
                is_true.reindex(idx_canon)[srcs].values.astype(float)]))

        Xp = design(canon)
        est = np.full(len(rows), np.nan)
        for k in np.unique(f):
            a, b = f != k, f == k
            if a.sum() < 30 or b.sum() == 0:
                continue
            est[b] = RidgeCV(alphas=alphas).fit(Xp[a], y[a]).predict(Xp[b])
        ok = np.isfinite(est)
        if ok.sum() < 30:
            continue

        blended = out[rows].copy()
        fold_ws = []
        for k in np.unique(f):
            a, b = (f != k) & ok, (f == k) & ok
            if a.sum() < 30 or b.sum() == 0:
                continue
            w, _ = physics.tune_weight(y[a], out[rows[a]], est[a])
            w *= shrink          # same bias argument as the physics blend
            fold_ws.append(w)
            blended[b] = physics.blend(out[rows[b]], est[b], w)

        gain = r2(y, blended) - r2(y, out[rows])
        info[t] = {"est_r2": r2(y[ok], est[ok]), "gain": gain}
        if gain <= 0:
            continue
        out[rows] = blended

        te = np.where((test_df["target_type"] == t).values)[0]
        if len(te):
            # The test weight MUST be the cross-fitted one. Re-tuning it here
            # against an in-sample Ridge fit (mdl.predict(Xp) on the very rows
            # mdl was fitted to) makes the estimate look far better than it is
            # out of sample, so the weight comes out inflated -- measured
            # eea 0.003 -> 0.250, ei 0.290 -> 0.550, eps 0.457 -> 0.650 -- and
            # that inflated weight is then applied to test predictions whose
            # estimate really is out of sample. It degrades only the test side,
            # so it is invisible in CV: a pure CV-to-leaderboard gap generator.
            w = float(np.mean(fold_ws)) if fold_ws else 0.0
            if w > 0:
                mdl = RidgeCV(alphas=alphas).fit(Xp[ok], y[ok])
                e_te = mdl.predict(design(test_df["canon"].values[te]))
                test_out[te] = physics.blend(test_out[te], e_te, w)

    if verbose and info:
        print(f"\n{'target':<7}{'partner-ridge R2':>18}{'blend gain':>13}")
        for t, dct in info.items():
            print(f"{t:<7}{dct['est_r2']:>18.4f}{dct['gain']:>+13.4f}")
    return out, test_out, info


# --------------------------------------------------------------------------- #
# local cache (LOCAL ONLY -- the exported notebook recomputes everything)      #
# --------------------------------------------------------------------------- #

def cache_get(key):
    p = cache_dir() / f"oof_{key}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def cache_put(key, **arrays):
    np.savez(cache_dir() / f"oof_{key}.npz", **arrays)


def report(train_df, oof, label=""):
    df = train_df.copy()
    df["pred"] = oof
    score, per = competition_score(df)
    print(f"\n=== {label} ===")
    for t in TARGETS:
        print(f"  {t:<4} R2 = {per[t]:+.4f}")
    print(f"  {'MEAN':<4}    = {score:+.4f}")
    return score, per
