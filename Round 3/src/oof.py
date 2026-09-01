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
    if n_rows > 1500:
        return max(2, min(base, 5))
    if n_rows >= 400:
        return base
    return max(base, 15)


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
                     use_partners=True, use_physics_feature=True):
    """OOF + test + whole-universe predictions for one per-property estimator.

    `universe` maps every canonical polymer to a predicted value for each target.
    Stage 2 of the physics blend needs it, because it must estimate a partner
    property for polymers that were never measured for it.

    Partner features (the measured values of a polymer's OTHER properties) are
    the strongest signal available for the DFT block: 88-99% of test rows in
    those targets have their polymer present in train under some property. They
    are built per fold from TRAINING rows only, and `drop_leaky` removes the
    column that would be the row's own label.
    """
    from src import partners as P

    oof = np.full(len(train_df), np.nan)
    test_pred = np.full(len(test_df), np.nan)
    universe = {t: np.full(len(all_canon), np.nan) for t in TARGETS}
    uni_df = pd.DataFrame({"canon": all_canon})

    for t in TARGETS:
        rows = np.where((train_df["target_type"] == t).values)[0]
        if len(rows) < 10:
            continue
        y_all = train_df["target"].values
        f = fold_id[rows]

        for k in np.unique(f):
            tr = rows[f != k]
            va = rows[f == k]
            if len(tr) < 5:
                continue
            if use_partners:
                # partner source = everything except the rows being predicted
                keep = np.ones(len(train_df), dtype=bool)
                keep[va] = False
                src = train_df[keep]
                (btr, bva), _ = P.build(src, [train_df.iloc[tr], train_df.iloc[va]])
                Xa = np.hstack([X_tr[tr], P.drop_leaky(btr, t).values])
                Xb = np.hstack([X_tr[va], P.drop_leaky(bva, t).values])
                if use_physics_feature:
                    ra, ha = physics_feature_cols(t, btr)
                    rb, hb = physics_feature_cols(t, bva)
                    if ra is not None:
                        ok = np.isfinite(ra) & ha
                        if ok.sum() >= 15 and np.std(ra[ok]) > 1e-12:
                            a_, b_ = np.polyfit(ra[ok], y_all[tr][ok], 1)
                            Xa = np.hstack([Xa, np.column_stack(
                                [a_ * ra + b_, ha.astype(np.float32)])])
                            Xb = np.hstack([Xb, np.column_stack(
                                [a_ * rb + b_, hb.astype(np.float32)])])
            else:
                Xa, Xb = X_tr[tr], X_tr[va]
            m = kind_maker(kind, t, len(tr), seed)
            m.fit(Xa, y_all[tr])
            oof[va] = m.predict(Xb)

        # full fit: partner source is all of train, guard still applied
        if use_partners:
            (bfull, bte, buni), _ = P.build(
                train_df, [train_df.iloc[rows], test_df, uni_df])
            Xfull = np.hstack([X_tr[rows], P.drop_leaky(bfull, t).values])
            Xtest = np.hstack([X_te, P.drop_leaky(bte, t).values])
            Xuni = np.hstack([X_all, P.drop_leaky(buni, t).values])
            if use_physics_feature:
                rf, hf = physics_feature_cols(t, bfull)
                rt, ht = physics_feature_cols(t, bte)
                ru, hu = physics_feature_cols(t, buni)
                if rf is not None:
                    ok = np.isfinite(rf) & hf
                    if ok.sum() >= 15 and np.std(rf[ok]) > 1e-12:
                        a_, b_ = np.polyfit(rf[ok], y_all[rows][ok], 1)
                        Xfull = np.hstack([Xfull, np.column_stack(
                            [a_ * rf + b_, hf.astype(np.float32)])])
                        Xtest = np.hstack([Xtest, np.column_stack(
                            [a_ * rt + b_, ht.astype(np.float32)])])
                        Xuni = np.hstack([Xuni, np.column_stack(
                            [a_ * ru + b_, hu.astype(np.float32)])])
        else:
            Xfull, Xtest, Xuni = X_tr[rows], X_te, X_all
        full = kind_maker(kind, t, len(rows), seed)
        full.fit(Xfull, y_all[rows])
        te = np.where((test_df["target_type"] == t).values)[0]
        if len(te):
            test_pred[te] = full.predict(Xtest[te])
        universe[t] = full.predict(Xuni)

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
    true_w = train_df.pivot_table(index="canon", columns="target_type",
                                  values="target", aggfunc="mean")
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


def apply_physics(train_df, oof, test_df, test_pred, fold_id, partners, is_true,
                  oof_by_target=None, n_rounds=3, damp=0.5, verbose=True):
    """Cross-fitted, staged, iterated physics blend.

    Three things are happening, and the third is where the leak lives if you are
    careless:

    1. STAGING. Rows are grouped by how many of the relation's sources are
       actually measured, and each group gets its own calibration and its own
       blend weight. A binary true/predicted split wastes the partially covered
       rows, and they are numerous -- of 148 `ei` test rows, 55 have both sources
       measured, 69 have exactly one.

    2. ITERATION. The seven properties form a constraint graph
       (ei = egc+eea, eea = ei-egc, egb ~ egc, eps ~ nc^2, nc ~ sqrt(eps)) that
       one pass of model predictions does not satisfy, so predicted cells are
       re-estimated from their neighbours and damped toward the estimate.

    3. THE MASK, which makes 1 and 2 honest. `eps` is refined FROM `nc`, and `nc`
       is then predicted FROM the refined `eps`. For a validation polymer whose
       `nc` is measured, its own label would flow into its own prediction -- a
       closed loop that reads as a huge gain and is entirely fake (it took the
       measured score from 0.893 to 0.935 before this mask existed). So for every
       (target, fold) the refinement is redone with that fold's true `target`
       values replaced by their out-of-fold predictions.

       The test side needs no such mask: only 2 of 4940 test rows have their
       (polymer, target_type) present in train, so a test row's own label is not
       in the table to leak.
    """
    oof = np.asarray(oof, dtype=float)
    oof_out = oof.copy()
    test_out = np.asarray(test_pred, dtype=float).copy()
    info = {}

    canon_pos = {c: i for i, c in enumerate(partners.index)}

    for t, (srcs, expr) in physics.RELATIONS.items():
        rows = np.where((train_df["target_type"] == t).values)[0]
        if len(rows) < 30:
            continue
        y = train_df["target"].values[rows]
        canon = train_df["canon"].values[rows]
        f = fold_id[rows]

        blended = oof_out[rows].copy()
        weights, levels = {}, np.zeros(len(rows), dtype=int)

        for k in np.unique(f):
            va = f == k
            if va.sum() == 0:
                continue
            # --- mask: hide this fold's own target labels, everywhere ---
            Pk = partners.copy()
            Ik = is_true.copy()
            pairs = [(canon_pos[c], v) for c, v in
                     zip(canon[va], oof[rows[va]]) if c in canon_pos]
            if pairs:
                pos = [i for i, _ in pairs]
                col = Pk[t].values.astype(float)
                col[pos] = [v for _, v in pairs]        # out-of-fold stand-in
                Pk[t] = col
                flag = Ik[t].values.copy()
                flag[pos] = False
                Ik[t] = flag
            Pk = refine_universe(Pk, Ik, train_df.iloc[np.setdiff1d(
                np.arange(len(train_df)), rows[va])], n_rounds=n_rounds, damp=damp)

            n_true_k = Ik.reindex(canon)[srcs].sum(axis=1).values.astype(int)
            levels[va] = n_true_k[va]
            for lvl in sorted(set(n_true_k.tolist()), reverse=True):
                tr_m = (n_true_k == lvl) & (~va)
                va_m = (n_true_k == lvl) & va
                if tr_m.sum() < 15 or va_m.sum() == 0:
                    continue
                rel = _fit_relation(train_df, rows[tr_m], t, srcs, expr, Pk)
                if rel is None:
                    continue
                e_tr = rel.apply(Pk.reindex(canon[tr_m])[srcs].values)
                e_va = rel.apply(Pk.reindex(canon[va_m])[srcs].values)
                w, _ = physics.tune_weight(y[tr_m], oof[rows[tr_m]], e_tr)
                blended[va_m] = physics.blend(oof[rows[va_m]], e_va, w)
                weights.setdefault(lvl, []).append(w)

        gain = r2(y, blended) - r2(y, oof[rows])
        info[t] = {"r2_before": r2(y, oof[rows]), "r2_after": r2(y, blended),
                   "gain": gain,
                   "n_by_level": {int(l): int((levels == l).sum())
                                  for l in sorted(set(levels.tolist()), reverse=True)},
                   "w_by_level": {int(l): round(float(np.mean(v)), 3)
                                  for l, v in sorted(weights.items(), reverse=True)}}
        if gain > 0:
            oof_out[rows] = blended

        # ---- test side: no mask needed, calibration on all train rows ----
        te = np.where((test_df["target_type"] == t).values)[0]
        if len(te) and gain > 0:
            Pfull = refine_universe(partners, is_true, train_df,
                                    n_rounds=n_rounds, damp=damp)
            tcanon = test_df["canon"].values[te]
            n_true_tr = is_true.reindex(canon)[srcs].sum(axis=1).values.astype(int)
            t_lvl = is_true.reindex(tcanon)[srcs].sum(axis=1).values.astype(int)
            for lvl in sorted(set(n_true_tr.tolist()), reverse=True):
                tr_m = n_true_tr == lvl
                te_m = t_lvl == lvl
                if tr_m.sum() < 15 or te_m.sum() == 0:
                    continue
                rel = _fit_relation(train_df, rows[tr_m], t, srcs, expr, Pfull)
                if rel is None:
                    continue
                e_tr = rel.apply(Pfull.reindex(canon[tr_m])[srcs].values)
                w, _ = physics.tune_weight(y[tr_m], oof[rows[tr_m]], e_tr)
                e_te = rel.apply(Pfull.reindex(tcanon[te_m])[srcs].values)
                test_out[te[te_m]] = physics.blend(test_out[te[te_m]], e_te, w)

    if verbose and info:
        print(f"\n{'target':<6}{'before':>9}{'after':>9}{'gain':>9}   "
              f"{'rows by #measured sources':<26}{'blend weight by level'}")
        for t, d in info.items():
            print(f"{t:<6}{d['r2_before']:>9.4f}{d['r2_after']:>9.4f}{d['gain']:>+9.4f}   "
                  f"{str(d['n_by_level']):<26}{d['w_by_level']}")
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


def partner_regression(train_df, pred, test_df, test_pred, fold_id, partners,
                       is_true, verbose=True):
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
        for k in np.unique(f):
            a, b = (f != k) & ok, (f == k) & ok
            if a.sum() < 30 or b.sum() == 0:
                continue
            w, _ = physics.tune_weight(y[a], out[rows[a]], est[a])
            blended[b] = physics.blend(out[rows[b]], est[b], w)

        gain = r2(y, blended) - r2(y, out[rows])
        info[t] = {"est_r2": r2(y[ok], est[ok]), "gain": gain}
        if gain <= 0:
            continue
        out[rows] = blended

        te = np.where((test_df["target_type"] == t).values)[0]
        if len(te):
            mdl = RidgeCV(alphas=alphas).fit(Xp[ok], y[ok])
            e_tr = mdl.predict(Xp)
            w, _ = physics.tune_weight(y, np.asarray(pred, dtype=float)[rows], e_tr)
            e_te = mdl.predict(design(test_df["canon"].values[te]))
            test_out[te] = physics.blend(test_out[te], e_te, w)

    if verbose and info:
        print(f"\n{'target':<7}{'partner-ridge R2':>18}{'blend gain':>13}")
        for t, dct in info.items():
            print(f"{t:<7}{dct['est_r2']:>18.4f}{dct['gain']:>+13.4f}")
    return out, test_out, info


def report(train_df, oof, label=""):
    df = train_df.copy()
    df["pred"] = oof
    score, per = competition_score(df)
    print(f"\n=== {label} ===")
    for t in TARGETS:
        print(f"  {t:<4} R2 = {per[t]:+.4f}")
    print(f"  {'MEAN':<4}    = {score:+.4f}")
    return score, per
