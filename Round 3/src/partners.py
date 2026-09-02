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


# ---------------------------------------------------------------------------
# Host-sanctioned Round-2 archive labels (see src/archive.py).
#
# A module-level registry rather than a threaded argument: `build()` is called
# from four places inside per_property_oof and partner_frame is called from
# run_ensemble, and one global that is set exactly once at load time is far
# easier to audit than five signatures that must all agree.
#
# SAFETY. These are extra values for `tg` and `egc` only. `drop_leaky()` still
# removes `true_{target}`/`has_{target}` for the row being predicted, and
# apply_physics masks the target's own partner column per fold, so an archive
# label can only ever enter as a SOURCE property, never as the row's own answer.
# `tg` is not in DFT_PROPS at all, so the sole feature effect is wider `true_egc`
# coverage -- which is exactly the coverage the test set has.
# ---------------------------------------------------------------------------
_ARCHIVE = None          # long format: canon, target_type, target


def set_archive(df) -> None:
    """Register (or clear, with None) the archive label table."""
    global _ARCHIVE
    _ARCHIVE = df


def get_archive():
    return _ARCHIVE


def label_pool(train_fold: pd.DataFrame) -> pd.DataFrame:
    """Training-fold labels plus the archive, long format.

    Duplicate (canon, target_type) keys are left in place; every consumer
    aggregates with mean, and the two sources agree where they overlap.
    """
    cols = ["canon", "target_type", "target"]
    base = train_fold[cols]
    if _ARCHIVE is None or len(_ARCHIVE) == 0:
        return base
    return pd.concat([base, _ARCHIVE[cols]], ignore_index=True)


def build(train_fold: pd.DataFrame, frames: list[pd.DataFrame]):
    """Return (list_of_feature_frames, column_names).

    `train_fold` supplies the labels. `frames` are the frames to featurise
    (typically [train_fold, valid_fold] or [train, test]).
    """
    wide = label_pool(train_fold).pivot_table(index="canon", columns="target_type",
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
