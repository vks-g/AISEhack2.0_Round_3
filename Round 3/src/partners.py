"""Co-observed partner features, with the leakage guard built in.

Many polymers carry more than one measured property. When predicting `ei` for a
polymer whose `egc` and `eea` were also measured, those measured values are
legitimate inputs -- they are in train.csv, and test polymers get the same
treatment. This is the single strongest feature family for the DFT block.

It is also the easiest place in the whole pipeline to leak. Two rules, both
enforced here rather than left to the caller:

  1. The partner table is built from the TRAINING FOLD ONLY. Building it from the
     full frame leaks validation labels into training and inflates CV.
  2. When predicting property P, the column `true_P` is dropped. It is that row's
     own label. `assert_no_leak()` proves both directions: that the leak is real
     if unguarded, and that the guard removes it.

Coverage is real but partial -- the fraction of each property's TEST rows whose
partners are present in train.csv is 35-62% (see src/physics). Rows without a
partner fall back to the training-fold mean, so the model must also work without.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.metric import TARGETS

# `tg` is measured on a disjoint set of polymers (only 4 of 4143 tg polymers
# carry any other property), so it is neither a useful partner nor helped by one.
DFT_PROPS = ["egc", "egb", "eps", "nc", "ei", "eea"]


def build(train_fold: pd.DataFrame, frames: list[pd.DataFrame]):
    """Return (list_of_feature_frames, column_names).

    `train_fold` supplies the labels. `frames` are the frames to featurise
    (typically [train_fold, valid_fold] or [train, test]).
    """
    wide = train_fold.pivot_table(index="canon", columns="target_type",
                                  values="target", aggfunc="mean")
    cols, fills = [], {}
    for p in DFT_PROPS:
        if p not in wide.columns:
            continue
        cols.append(f"true_{p}")
        fills[f"true_{p}"] = float(wide[p].mean())

    out = []
    for df in frames:
        block = pd.DataFrame(index=range(len(df)))
        for p in DFT_PROPS:
            if f"true_{p}" not in cols:
                continue
            v = wide[p].reindex(df["canon"].values).values
            c = f"true_{p}"
            block[c] = np.where(np.isfinite(v), v, fills[c])
            block[f"has_{p}"] = np.isfinite(v).astype(np.float32)
        block["n_partners"] = block[[f"has_{p}" for p in DFT_PROPS
                                     if f"has_{p}" in block]].sum(axis=1)
        out.append(block.astype(np.float32))
    names = list(out[0].columns) if out else []
    return out, names


def drop_leaky(block: pd.DataFrame, target_type: str) -> pd.DataFrame:
    """Remove the columns that encode this row's own label."""
    banned = {f"true_{target_type}", f"has_{target_type}"}
    return block[[c for c in block.columns if c not in banned]]


def assert_no_leak(train: pd.DataFrame) -> None:
    """Prove the leak exists, then prove drop_leaky() removes it.

    A guard that passes without a demonstrable leak proves nothing, so this
    checks both directions and raises on either failure.
    """
    blocks, _ = build(train, [train])
    block = blocks[0]
    for p in DFT_PROPS:
        m = (train["target_type"] == p).values
        if m.sum() == 0 or f"true_{p}" not in block:
            continue
        got = block.loc[m, f"true_{p}"].values
        want = train.loc[m, "target"].values
        if not np.allclose(got, want, atol=1e-6):
            raise AssertionError(
                f"true_{p} does not reproduce the {p} target -- partner build is wrong")
        if f"true_{p}" in drop_leaky(block.loc[m], p).columns:
            raise AssertionError(f"drop_leaky failed to remove true_{p}")
    print(f"partner leakage guard OK ({len(DFT_PROPS)} properties: leak demonstrated, guard removes it)")
