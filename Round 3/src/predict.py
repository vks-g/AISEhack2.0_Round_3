"""Full fit on all of train, inference on test, write submission.csv.

    ./.venv/bin/python -m src.predict --config lgbm --out submission.csv

This is the same `fit` / `predict` pair that `src.cv` scores, called once on the
whole training set. There is no second implementation of the model, so what CV
measured is what gets submitted.

Predictions are clipped to each property's observed training range plus 5%: a
negative bandgap is not a polymer, and a clip is the cheapest guard against a
single wild extrapolation wrecking an R2 that averages over only ~150 test rows.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.cv import load_config
from src.data import load_test, load_train
from src.metric import TARGETS


def build(config_name: str, out: str = "submission.csv", seed: int = 42,
          clip: bool = True) -> pd.DataFrame:
    cfg = load_config(config_name)
    train, test = load_train(), load_test()
    print(f"train {len(train)} rows / {train.canon.nunique()} polymers   "
          f"test {len(test)} rows / {test.canon.nunique()} polymers")

    t0 = time.time()
    state = cfg.fit(train, seed=seed)
    print(f"fit   {time.time() - t0:.0f}s")
    pred = np.asarray(cfg.predict(state, test), dtype=float)
    print(f"total {time.time() - t0:.0f}s")

    if clip:
        for t in TARGETS:
            m = (test["target_type"] == t).values
            if not m.any():
                continue
            v = train.loc[train.target_type == t, "target"]
            lo, hi = v.min(), v.max()
            pad = 0.05 * (hi - lo)
            n_clipped = int(((pred[m] < lo - pad) | (pred[m] > hi + pad)).sum())
            pred[m] = np.clip(pred[m], lo - pad, hi + pad)
            if n_clipped:
                print(f"  {t}: clipped {n_clipped}/{m.sum()} to [{lo-pad:.4g}, {hi+pad:.4g}]")

    bad = ~np.isfinite(pred)
    if bad.any():
        print(f"WARNING: {bad.sum()} non-finite predictions -> per-target mean")
        for t in TARGETS:
            m = (test["target_type"] == t).values & bad
            if m.any():
                pred[m] = float(train.loc[train.target_type == t, "target"].mean())

    sub = pd.DataFrame({"id": test["id"].values, "target": pred})
    sub.to_csv(out, index=False)
    print(f"\nwrote {out}: {len(sub)} rows")
    print(sub.head().to_string(index=False))
    return sub


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-clip", action="store_true")
    a = p.parse_args()
    build(a.config, out=a.out, seed=a.seed, clip=not a.no_clip)
