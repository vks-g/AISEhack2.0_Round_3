"""End-to-end ensemble: base models -> cross-fitted stack -> two-stage physics.

    ./.venv/bin/python -m src.run_ensemble --models lgbm,xgb,cb
    ./.venv/bin/python -m src.run_ensemble --models lgbm,xgb,cb,mtnn,cnn --out submission.csv

This is the scoring path AND the submission path, so what it prints is what the
exported notebook reproduces. Base-model OOF is cached locally under .cache/ so
stacking and physics can be iterated without refitting; the cache is LOCAL ONLY
and the exported notebook always recomputes (rule 6.2.4).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src import oof as O
from src.data import load_test, load_train
from src.features import featurize
from src.metric import TARGETS, competition_score
from src.paths import REPO

TREE_KINDS = {"lgbm", "xgb", "cb"}
SIMPLE_KINDS = {"knn", "et", "ridge"}
GLOBAL_MODELS = {"mtnn", "cnn", "gnn"}


def main(models, n_folds=10, seed=42, nn_seeds=(42, 202, 777), out=None,
         use_cache=True, verbose=True, scheme='per_property', use_partners=True,
         use_physics_feature=True, use_partner_ridge=True):
    t0 = time.time()
    train, test = load_train(), load_test()
    all_canon = list(dict.fromkeys(list(train["canon"]) + list(test["canon"])))
    print(f"train {len(train)}  test {len(test)}  universe {len(all_canon)} polymers")

    X_tr = featurize(train["canon"])
    X_te = featurize(test["canon"])
    X_all = featurize(all_canon)
    print(f"features {X_tr.shape[1]}  ({time.time()-t0:.0f}s)")

    fold_id = O.build_fold_id(train, n_folds=n_folds, seed=seed, scheme=scheme)
    print(f"folds: {n_folds} x {scheme}   partners: {use_partners}   physics-feature: {use_physics_feature}   partner-ridge: {use_partner_ridge}")

    base_oof, base_test, tree_universe = {}, {}, {}

    for name in models:
        key = f"{name}_f{n_folds}_s{seed}_{scheme}_p{int(use_partners)}_pf{int(use_physics_feature)}_pr{int(use_partner_ridge)}v4"
        hit = O.cache_get(key) if use_cache else None
        if hit is not None:
            base_oof[name], base_test[name] = hit["oof"], hit["test"]
            if "universe" in hit:
                tree_universe[name] = hit["universe"]
            print(f"  {name:<6} cached")
            continue

        t1 = time.time()
        if name in TREE_KINDS or name in SIMPLE_KINDS:
            if name in TREE_KINDS:
                from src.models import trees as _mod
            else:
                from src.models import simple as _mod
            o, te, uni = O.per_property_oof(_mod.make, name, train, X_tr,
                                            test, X_te, fold_id, all_canon,
                                            X_all, seed=seed,
                                            use_partners=use_partners,
                                            use_physics_feature=use_physics_feature,
                                            use_partner_ridge=use_partner_ridge)
            uni_arr = np.column_stack([uni[t] for t in TARGETS])
            tree_universe[name] = uni_arr
            if use_cache:
                O.cache_put(key, oof=o, test=te, universe=uni_arr)
        elif name == "mtnn":
            from src.models import mtnn
            o, te = mtnn.oof_and_test(train, X_tr, test, X_te, fold_id,
                                      list(nn_seeds))
            if use_cache:
                O.cache_put(key, oof=o, test=te)
        elif name == "cnn":
            from src.models import cnn
            o, te = cnn.oof_and_test(train, test, fold_id, list(nn_seeds))
            if use_cache:
                O.cache_put(key, oof=o, test=te)
        elif name == "gnn":
            from src.models import gnn
            o, te = gnn.oof_and_test(train, test, fold_id, list(nn_seeds))
            if use_cache:
                O.cache_put(key, oof=o, test=te)
        else:
            raise ValueError(f"unknown model {name}")

        base_oof[name], base_test[name] = np.asarray(o), np.asarray(te)
        df = train.copy(); df["pred"] = base_oof[name]
        s, _ = competition_score(df)
        print(f"  {name:<6} mean R2 {s:+.4f}   ({time.time()-t1:.0f}s)")

    # ---- per-model report ----
    if verbose:
        print(f"\n{'model':<8}" + "".join(f"{t:>8}" for t in TARGETS) + f"{'MEAN':>9}")
        for name in models:
            df = train.copy(); df["pred"] = base_oof[name]
            s, per = competition_score(df)
            print(f"{name:<8}" + "".join(f"{per[t]:>8.4f}" for t in TARGETS)
                  + f"{s:>9.4f}")

    # ---- stack ----
    st_oof, st_test, alphas = O.stack(train, base_oof, base_test, test, fold_id)
    score_stack, _ = O.report(train, st_oof, "stacked")

    # ---- physics ----
    if not tree_universe:
        print("\nno tree universe available -- skipping physics")
        return {"mean_r2": score_stack, "oof": st_oof, "test": st_test}

    # The universe that fills MISSING partner values must come from models that
    # never saw a true label as a feature. A partner-fed model's prediction for
    # polymer p on property s is a function of p's true value for every OTHER
    # property -- including the one we are about to predict from it. That closes
    # a one-hop cycle and inflated OOF by +0.008 (0.9097 -> 0.9014 once removed)
    # while contributing nothing on test, where no such labels exist.
    ck = f"cleanuni_f{n_folds}_s{seed}v4"
    hit = O.cache_get(ck) if use_cache else None
    if hit is not None:
        uni = hit["universe"]
        print("  clean universe: cached")
    else:
        from src.models import trees as _tr
        t1 = time.time()
        _, _, cu = O.per_property_oof(_tr.make, "lgbm", train, X_tr, test, X_te,
                                      fold_id, all_canon, X_all, seed=seed,
                                      use_partners=False, use_physics_feature=False,
                                      use_partner_ridge=False)
        uni = np.column_stack([cu[t] for t in TARGETS])
        if use_cache:
            O.cache_put(ck, universe=uni)
        print(f"  clean universe built ({time.time()-t1:.0f}s)")
    universe_by_target = {t: uni[:, i] for i, t in enumerate(TARGETS)}
    partners, is_true = O.partner_frame(train, all_canon, universe_by_target)

    ion = O.ionic_term(train, X_tr, all_canon, X_all, fold_id, seed=seed)
    ph_oof, ph_test, info = O.apply_physics(train, st_oof, test, st_test,
                                            fold_id, partners, is_true,
                                            n_rounds=0, ionic=ion)
    O.report(train, ph_oof, "stacked + staged physics")
    ph_oof, ph_test, _ = O.partner_regression(train, ph_oof, test, ph_test,
                                              fold_id, partners, is_true)
    score, per = O.report(train, ph_oof, "+ generalized partner regression")
    print(f"\nwall {time.time()-t0:.0f}s")

    if out:
        final = np.asarray(ph_test, dtype=float)
        for t in TARGETS:
            m = (test["target_type"] == t).values
            if not m.any():
                continue
            v = train.loc[train.target_type == t, "target"]
            lo, hi = v.min(), v.max()
            pad = 0.05 * (hi - lo)
            final[m] = np.clip(final[m], lo - pad, hi + pad)
        bad = ~np.isfinite(final)
        if bad.any():
            for t in TARGETS:
                m = (test["target_type"] == t).values & bad
                if m.any():
                    final[m] = float(train.loc[train.target_type == t, "target"].mean())
        pd.DataFrame({"id": test["id"].values, "target": final}).to_csv(out, index=False)
        print(f"wrote {out}  ({len(test)} rows)")

    return {"mean_r2": score, "per_target": per, "oof": ph_oof, "test": ph_test,
            "physics": info, "stack_alphas": alphas, "stack_only": score_stack}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="lgbm,xgb,cb")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nn-seeds", default="42,202,777")
    p.add_argument("--out", default=None)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--scheme", default="per_property", choices=["per_property", "grouped"])
    p.add_argument("--no-partners", action="store_true")
    p.add_argument("--no-physics-feature", action="store_true")
    a = p.parse_args()
    main([m.strip() for m in a.models.split(",") if m.strip()],
         n_folds=a.folds, seed=a.seed,
         nn_seeds=tuple(int(s) for s in a.nn_seeds.split(",")),
         out=a.out, use_cache=not a.no_cache, scheme=a.scheme,
         use_partners=not a.no_partners,
         use_physics_feature=not a.no_physics_feature)
