"""Path resolution that works identically locally and inside a Kaggle notebook."""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parent

_KAGGLE_CANDIDATES = [
    Path("/kaggle/input/aisehack-2-0"),
    Path("/kaggle/input"),
]


def data_dir() -> Path:
    """Directory holding train.csv / test.csv."""
    env = os.environ.get("AISE_DATA_DIR")
    if env:
        return Path(env)
    local = REPO / "data"
    if (local / "train.csv").exists():
        return local
    for c in _KAGGLE_CANDIDATES:
        if (c / "train.csv").exists():
            return c
        if c.exists():
            for sub in sorted(c.iterdir()):
                if (sub / "train.csv").exists():
                    return sub
    raise FileNotFoundError("could not locate train.csv; set AISE_DATA_DIR")


def cache_dir() -> Path:
    """LOCAL-ONLY scratch for the feature cache.

    Never referenced by exported notebooks. Competition rule 6.2.4 forbids
    shipping cached features; this exists purely so local CV iterates in seconds.
    """
    d = REPO / ".cache"
    d.mkdir(exist_ok=True)
    return d


DATA = data_dir
TRAIN_CSV = lambda: data_dir() / "train.csv"        # noqa: E731
TEST_CSV = lambda: data_dir() / "test.csv"          # noqa: E731
