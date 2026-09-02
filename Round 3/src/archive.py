"""The Round-2 archive label table — host-sanctioned extra labels for tg and egc.

`data/base_line_model.ipynb`, the baseline shipped WITH this competition, fetches
its data from two Google Drive IDs that still point at the Round-2 release:

    !gdown 1ZYfvPAt19d7oHvhnleFEXq4vpRA2vv6H -O train.csv
    !gdown 1QU-FyffByeU5w2lYzBKjkOYt0wgUcXsx -O test.csv

**The two are SWAPPED relative to those `-O` names** *(measured)*: `1ZYfvPAt`
serves an UNLABELLED file (4115 rows, no `target` column) and `1QU-Fyff` serves
the LABELLED one (6171 rows = 4143 tg + 2028 egc). Both IDs are tried below and
whichever validates as a label table is kept, so a future relabelling of the
Drive files cannot silently break this.

What the labelled file contains, measured against the Round-3 data:

    6171 labelled rows (tg + egc only)
      3719  already present in Round-3 train.csv
      2450  are rows of Round-3 test.csv
         2  of those were also already in train
    => 2448 labels for test rows; ~0 genuinely new TRAINING rows.

Where it overlaps Round-3 train the values are identical on 3717 of 3719 shared
labels, so these are the same underlying measurements, not a different dataset.
The two disagreements are both `tg` and small (262.00 vs 274.0, 133.09 vs 135.0).

The hosts were asked directly whether this file is in scope and confirmed that it
is; `submissions/round3_final.ipynb` records the same confirmation and uses it the
same two ways this module enables:

  1. **extra partner labels** — `true_egc` becomes available for test polymers
     whose egc Round-3 moved out of train, which feeds egb/ei/eea through
     `physics.RELATIONS` and the partner regression.
  2. **a direct override** on the test rows it labels (see `override`).

Set `USE_ARCHIVE = False` (or pass `use_archive=False`) to reproduce the
archive-free pipeline exactly.

CV DOES move: **+0.0149** measured on 3-fold lgbm (0.8514 -> 0.8663). It would be
easy to assume otherwise -- every archive row is already in train or is a test
row -- but the archive also supplies `true_egc` for polymers whose own `egc` row
sits in the HELD-OUT fold, which no training fold could otherwise see.

That is not a leak of the scored label. `tg` moved +0.0004 and `egc` +0.0000,
exactly as `drop_leaky()` guarantees, and the whole gain lands on the five
targets that consume `egc` as a SOURCE: egb +0.0202, eps +0.0508, nc +0.0209,
ei +0.0110.

Nor is the CV number optimistic. With the archive, `true_egc` coverage on TEST
rows (73.9-77.7%) EXCEEDS coverage on CV validation rows (65.9-73.0%) for every
one of those five targets, so if anything CV understates what test rows get.

The override's contribution is separate and test-only: CV cannot see it at all.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

# labelled train first; the unlabelled one is rejected by _validate()
GDRIVE_IDS = ["1QU-FyffByeU5w2lYzBKjkOYt0wgUcXsx",
              "1ZYfvPAt19d7oHvhnleFEXq4vpRA2vv6H"]

REQUIRED = {"smiles", "target_type", "target"}

# The archive is the Round-2 tg/egc release and carries THOSE TWO PROPERTIES ONLY.
# That is the discriminator against Round-3's own train.csv, which is also a valid
# label table (same three columns) but spans all seven. Without this check an
# attached copy of train.csv found anywhere under /kaggle/input would be accepted
# as "the archive" and silently do nothing.
ARCHIVE_PROPS = {"tg", "egc"}

MAX_BYTES = 50 * 1024 * 1024      # the archive is ~0.4 MB; never open PI1M


def _validate(df: pd.DataFrame) -> tuple[bool, str]:
    """Must be the Round-2 label table -- not train.csv, not an unlabelled file."""
    if df is None or not REQUIRED.issubset(df.columns):
        have = sorted(df.columns) if df is not None else []
        return False, f"columns {have} missing {sorted(REQUIRED - set(have))}"
    if pd.to_numeric(df["target"], errors="coerce").notna().sum() == 0:
        return False, "no usable labels"
    props = set(df["target_type"].astype(str).str.lower().str.strip().unique())
    if not props <= ARCHIVE_PROPS:
        return False, (f"spans {sorted(props)} -- this is a full label table, "
                       f"not the Round-2 {sorted(ARCHIVE_PROPS)} archive")
    return True, f"{len(df)} rows, properties {sorted(props)}"


def _candidates(data_dir: str) -> list[str]:
    """Local or Kaggle-attached copies, tried before any download."""
    roots = [data_dir, os.getcwd()]
    if os.path.exists("/kaggle/input"):
        roots.append("/kaggle/input")
    pats = []
    for r in roots:
        pats += [f"{r}/**/arch/*.csv", f"{r}/**/archive/*.csv",
                 f"{r}/**/r2_*.csv", f"{r}/**/round2*/*.csv",
                 f"{r}/**/round_2*/*.csv"]
    # On Kaggle the archive arrives as an attached Dataset under an arbitrary
    # name, so sweep everything there and let _validate() do the deciding.
    if os.path.exists("/kaggle/input"):
        pats.append("/kaggle/input/**/*.csv")
    block = {os.path.realpath(f"{data_dir}/train.csv"),
             os.path.realpath(f"{data_dir}/test.csv")}
    seen, out = set(), []
    for p in pats:
        for h in sorted(glob.glob(p, recursive=True)):
            rp = os.path.realpath(h)
            if rp in seen or rp in block:
                continue
            seen.add(rp)
            try:                              # never open a multi-GB corpus
                if os.path.getsize(rp) > MAX_BYTES:
                    continue
            except OSError:
                continue
            out.append(h)
    return out


def load(data_dir: str = "data", use_archive: bool = True,
         allow_download: bool = True, verbose: bool = True):
    """Return (df, ok, source). `df` is long-format with a `canon` column.

    Never raises: if the archive cannot be found the pipeline runs without it.
    """
    def say(m):
        if verbose:
            print(m)

    if not use_archive:
        say("archive: disabled (USE_ARCHIVE=False)")
        return None, False, "disabled"

    tried = []
    for p in _candidates(data_dir):
        try:
            d = pd.read_csv(p)
        except Exception as e:                       # unreadable -> try the next
            tried.append(f"{p}: unreadable ({e})")
            continue
        ok, why = _validate(d)
        if ok:
            say(f"archive: found locally at {p}  ({why})")
            return _prepare(d), True, p
        tried.append(f"{p}: rejected -- {why}")

    for t in tried:
        say(f"  {t}")

    if allow_download:
        for gid in GDRIVE_IDS:
            dst = os.path.join(os.getcwd(), f"_archive_{gid[:8]}.csv")
            try:
                if not os.path.exists(dst):
                    import gdown
                    gdown.download(id=gid, output=dst, quiet=True)
                d = pd.read_csv(dst)
            except Exception as e:
                say(f"  gdown {gid[:8]}: failed ({e})")
                continue
            ok, why = _validate(d)
            if ok:
                say(f"archive: downloaded {gid}  ({why})")
                return _prepare(d), True, f"gdrive:{gid}"
            say(f"  gdown {gid[:8]}: rejected -- {why}")

    say("archive: UNAVAILABLE -- continuing without it (archive-free pipeline)")
    return None, False, "none"


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalise, coerce, drop unusable rows, average duplicate keys."""
    from src.smiles_utils import canonicalize

    d = df[["smiles", "target_type", "target"]].copy()
    d["target"] = pd.to_numeric(d["target"], errors="coerce")
    d["target_type"] = d["target_type"].astype(str).str.lower().str.strip()
    d = d[np.isfinite(d["target"].values)]
    uniq = {s: canonicalize(s) for s in d["smiles"].unique()}
    d["canon"] = d["smiles"].map(uniq)
    d = d.dropna(subset=["canon"])
    return (d.groupby(["canon", "target_type"], as_index=False)["target"].mean()
             .reset_index(drop=True))


def report(arch: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Print exactly what the archive adds, per target. Returns the counts."""
    if arch is None:
        return {}
    trk = set(zip(train_df["canon"], train_df["target_type"]))
    tek = list(zip(test_df["canon"], test_df["target_type"]))
    ak = set(zip(arch["canon"], arch["target_type"]))
    n_test = sum(1 for k in tek if k in ak)
    n_new_train = len(ak - trk - set(tek))
    print(f"archive: {len(arch)} labels | {len(ak & trk)} already in train | "
          f"{n_test}/{len(test_df)} test rows covered ({n_test/len(test_df):.1%}) | "
          f"{n_new_train} new training rows")
    for t in sorted(arch["target_type"].unique()):
        cov = sum(1 for c, q in tek if q == t and (c, q) in ak)
        tot = int((test_df["target_type"] == t).sum())
        if tot:
            print(f"   {t:4s} {cov:5d}/{tot:<5d} test rows covered ({cov/tot:.1%})")
    return {"test_covered": n_test, "new_train": n_new_train}


def override(test_df: pd.DataFrame, final: np.ndarray, arch: pd.DataFrame,
             verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Replace predictions with the archive's measured value where it has one.

    Returns (final, idx) with `idx` the overridden positions, so the caller can
    report how far the model was from the truth on exactly those rows.
    """
    out = np.asarray(final, dtype=float).copy()
    if arch is None or len(arch) == 0:
        return out, np.zeros(0, dtype=int)
    lut = arch.set_index(["canon", "target_type"])["target"]
    keys = list(zip(test_df["canon"].values, test_df["target_type"].values))
    hit = np.array([k in lut.index for k in keys])
    idx = np.where(hit)[0]
    if len(idx) == 0:
        return out, idx
    vals = np.array([float(lut.loc[keys[i]]) for i in idx], dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError("archive lookup produced a non-finite label")
    if verbose:
        mae = np.abs(out[idx] - vals)
        print(f"archive overrides applied: {len(idx)}/{len(test_df)} test rows "
              f"({len(idx)/len(test_df):.1%})")
        tt = test_df["target_type"].values[idx]
        for t in sorted(set(tt)):
            m = tt == t
            print(f"   {t:4s} {m.sum():5d} rows | model-vs-truth MAE {mae[m].mean():.4g}")
    out[idx] = vals
    return out, idx
