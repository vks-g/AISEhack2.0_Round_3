"""Loading + canonicalisation. Single entry point so every config sees the same frame."""
from __future__ import annotations

import pandas as pd

from src.metric import TARGETS
from src.paths import TEST_CSV, TRAIN_CSV
from src.smiles_utils import canonicalize


def load_train(dedupe: bool = True) -> pd.DataFrame:
    """train.csv + a `canon` column.

    3 (smiles, target_type) keys in train.csv carry two conflicting targets
    (all `tg`: 244/239, 61.1/72.08, 105.0/98.28). With dedupe=True they are
    averaged, giving 7405 rows from 7409.
    """
    df = pd.read_csv(TRAIN_CSV())
    missing = {"smiles", "target", "target_type"} - set(df.columns)
    if missing:
        raise ValueError(f"train.csv missing columns {missing}")
    df["canon"] = df["smiles"].map(canonicalize)
    _assert_targets(df)
    if dedupe:
        df = (df.groupby(["canon", "target_type"], as_index=False)
                .agg(target=("target", "mean"), smiles=("smiles", "first")))
    return df.reset_index(drop=True)


def load_test() -> pd.DataFrame:
    df = pd.read_csv(TEST_CSV())
    missing = {"id", "smiles", "target_type"} - set(df.columns)
    if missing:
        raise ValueError(f"test.csv missing columns {missing}")
    df["canon"] = df["smiles"].map(canonicalize)
    _assert_targets(df)
    return df.reset_index(drop=True)


def _assert_targets(df: pd.DataFrame) -> None:
    unknown = set(df["target_type"].unique()) - set(TARGETS)
    if unknown:
        raise ValueError(
            f"unexpected target_type {sorted(unknown)}; expected lowercase {TARGETS}"
        )
