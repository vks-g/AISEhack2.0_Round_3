"""The scoring harness. This number, and only this number, decides a submission.

    ./.venv/bin/python -m src.cv --config lgbm
    ./.venv/bin/python -m src.cv --config lgbm --seed 7 --folds 5

A config module exposes exactly two functions:

    fit(train_df, seed, targets=None) -> state
    predict(state, df)                -> np.ndarray aligned to df's rows

`targets` lets CV fit only the property it is about to score instead of all
seven; pass None (the default) for the full fit that produces a submission.

Everything else -- cross-validation, the full fit for inference, the invariance
audit, the explainability report -- is derived from those two, so there is one
implementation of the model and no way for CV and submission to drift apart.
"""
from __future__ import annotations

import argparse
import importlib
import json
import time

import numpy as np
import pandas as pd

from src.data import load_train
from src.metric import TARGETS, competition_score, report
from src.paths import REPO
from src.splits import n_folds_for


def load_config(name: str):
    cfg = importlib.import_module(f"src.configs.{name}")
    for fn in ("fit", "predict"):
        if not hasattr(cfg, fn):
            raise AttributeError(
                f"config '{name}' must define {fn}(); see src/configs/lgbm.py")
    return cfg


def cross_validate(cfg, df: pd.DataFrame, seed: int = 42,
                   folds: int | None = None, verbose: bool = True):
    """Per-property CV. Returns (oof_predictions, per_fold_scores).

    Per-property KFold is leakage-safe here: within one target_type there are
    essentially no duplicate canonical polymers (see src/splits).
    """
    from sklearn.model_selection import KFold

    oof = np.full(len(df), np.nan)
    fold_scores: dict[str, list[float]] = {}

    for t in TARGETS:
        rows = np.where((df["target_type"] == t).values)[0]
        if len(rows) == 0:
            continue
        k = folds or n_folds_for(len(rows))
        k = max(2, min(k, len(rows)))
        sub = df.iloc[rows].reset_index(drop=True)
        scores = []
        for tr, va in KFold(k, shuffle=True, random_state=seed).split(rows):
            # A property's model may still want the OTHER properties' rows for
            # partner features, so hand the config the full training frame minus
            # this fold's validation polymers.
            held = set(sub.iloc[va]["canon"])
            mask = ~((df["canon"].isin(held)) & (df["target_type"] == t))
            state = cfg.fit(df[mask].reset_index(drop=True), seed=seed, targets=[t])
            p = cfg.predict(state, sub.iloc[va].reset_index(drop=True))
            oof[rows[va]] = p
            from src.metric import r2
            scores.append(r2(sub.iloc[va]["target"].values, p))
        fold_scores[t] = scores
        if verbose:
            print(f"  {t:<4} n={len(rows):<5} folds={k:<3} "
                  f"R2={np.mean(scores):+.4f} +/- {np.std(scores):.4f}")
    return oof, fold_scores


def run(config_name: str, seed: int = 42, folds: int | None = None,
        save: bool = True) -> dict:
    cfg = load_config(config_name)
    df = load_train()
    print(f"rows={len(df)}  unique polymers={df['canon'].nunique()}")
    print(f"config={config_name}  seed={seed}  "
          f"folds={'adaptive' if folds is None else folds}\n")

    t0 = time.time()
    oof, fold_scores = cross_validate(cfg, df, seed=seed, folds=folds)
    wall = time.time() - t0

    df = df.copy()
    df["pred"] = oof
    print(f"\n=== {config_name} ===")
    score, per = report(df)

    # Noise floor for the MEAN score. Each target's OOF R2 has a standard error
    # of roughly std(fold scores)/sqrt(n_folds); the mean of seven of them has
    # SE = sqrt(sum of squares)/7. Deltas under ~2x this are not improvements.
    per_target_std = {t: float(np.std(v)) for t, v in fold_scores.items()}
    ses = [s / np.sqrt(max(len(fold_scores[t]), 1)) for t, s in per_target_std.items()]
    noise = float(np.sqrt(sum(x * x for x in ses)) / len(TARGETS))
    print(f"\nwall {wall:.0f}s   noise floor ~{noise:.4f} "
          f"(treat deltas below {2*noise:.4f} as no change)")
    worst = max(per_target_std, key=per_target_std.get)
    print(f"noisiest target: {worst} (fold std {per_target_std[worst]:.3f}) -- "
          f"confirm any win there with a second seed")

    result = {
        "config": config_name, "seed": seed, "mean_r2": score,
        "per_target": per, "per_target_fold_std": per_target_std,
        "noise_floor": noise, "wall_seconds": round(wall, 1),
    }
    if save:
        out = REPO / "experiments" / "runs"
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (out / f"{stamp}_{config_name}_seed{seed}.json").write_text(
            json.dumps(result, indent=2))
        np.save(out / f"{stamp}_{config_name}_seed{seed}_oof.npy", oof)
        print(f"saved -> experiments/runs/{stamp}_{config_name}_seed{seed}.json")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=None,
                   help="override the adaptive fold count (10, or 15 when n<400)")
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args()
    run(a.config, seed=a.seed, folds=a.folds, save=not a.no_save)
