"""Submission validator. Run this before every single upload.

    ./.venv/bin/python -m src.check_submission submission.csv

Every threshold is derived from test.csv / train.csv at run time. Nothing is
hard-coded, because the hard-coded "4497 rows" that was previously carried in
this repo's docs is wrong -- test.csv has 4940 rows; 4497 is its unique-SMILES
count.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src.data import load_test, load_train
from src.metric import TARGETS


def check(path: str) -> bool:
    test = load_test()
    train = load_train(dedupe=False)
    fails, warns = [], []

    try:
        sub = pd.read_csv(path)
    except Exception as e:
        print(f"FAIL  cannot read {path}: {e}")
        return False

    if list(sub.columns) != ["id", "target"]:
        fails.append(f"columns are {list(sub.columns)}, expected exactly ['id', 'target']")
    if len(sub) != len(test):
        fails.append(f"{len(sub)} rows, test.csv has {len(test)}")

    if "id" in sub.columns:
        missing = set(test["id"]) - set(sub["id"])
        extra = set(sub["id"]) - set(test["id"])
        if missing:
            fails.append(f"{len(missing)} test ids missing (e.g. {sorted(missing)[:5]})")
        if extra:
            fails.append(f"{len(extra)} ids not in test.csv (e.g. {sorted(extra)[:5]})")
        if sub["id"].duplicated().any():
            fails.append(f"{int(sub['id'].duplicated().sum())} duplicate ids")

    if "target" in sub.columns:
        v = pd.to_numeric(sub["target"], errors="coerce")
        if v.isna().any():
            fails.append(f"{int(v.isna().sum())} non-numeric or NaN targets")
        elif not np.isfinite(v).all():
            fails.append(f"{int((~np.isfinite(v)).sum())} inf targets")
        else:
            merged = test.merge(sub, on="id", how="left")
            for t in TARGETS:
                m = (merged["target_type"] == t).values
                if not m.any():
                    continue
                got = merged.loc[m, "target"].values
                obs = train.loc[train.target_type == t, "target"]
                lo, hi = obs.min(), obs.max()
                pad = 0.25 * (hi - lo)
                out = ((got < lo - pad) | (got > hi + pad)).sum()
                if out:
                    warns.append(f"{t}: {out}/{m.sum()} predictions outside "
                                 f"[{lo - pad:.3g}, {hi + pad:.3g}] (train range "
                                 f"[{lo:.3g}, {hi:.3g}])")
                if np.std(got) < 1e-8:
                    fails.append(f"{t}: all {m.sum()} predictions identical "
                                 f"({got[0]:.4g}) -- target mapping is probably broken")
                print(f"  {t:<4} n={int(m.sum()):<5} pred [{got.min():9.4g}, {got.max():9.4g}] "
                      f"mean {got.mean():9.4g}   train [{lo:9.4g}, {hi:9.4g}]")

    print()
    for w in warns:
        print(f"WARN  {w}")
    for f in fails:
        print(f"FAIL  {f}")
    if not fails:
        print(f"PASS  {path}: {len(sub)} rows, ids match test.csv, all finite, "
              f"per-target ranges plausible")
    return not fails


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", default="submission.csv")
    a = p.parse_args()
    sys.exit(0 if check(a.path) else 1)
