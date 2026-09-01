"""Physics relations between the DFT properties.

The four relations, MEASURED on co-observed train pairs. `raw` is the textbook
identity; `fitted` is the same expression passed through a 1-D least-squares
calibration a*x + b estimated on the training co-observations:

    target  expression        n     raw R2    fitted R2   a       b
    ei      egc + eea         59    0.9629    0.9650      1.005    0.013
    eea     ei  - egc         59    0.9710    0.9727      1.004   -0.054
    egb     egc              175    0.8922    0.9282      1.159   -1.044
    eps     nc ** 2          134    0.3364    0.8553      1.040    0.615
    nc      sqrt(eps)        134    0.1708    0.8370      0.887    0.050

Read the eps and nc rows carefully. The Maxwell relation eps = n^2 holds for the
optical dielectric constant; the measured static eps sits well above it, so the
RAW identity is badly biased (R2 0.34) while the SAME expression with a fitted
offset is worth 0.86. Both shipped notebooks apply the raw form. Always fit.

Coverage on test -- the fraction of a property's test rows whose partner values
are present in train.csv, i.e. the rows this can actually touch with TRUE inputs:

    ei 55/148 (37%)   eea 51/147 (35%)   egb 124/224 (55%)
    eps 95/153 (62%)  nc 95/153 (62%)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Relation:
    target: str
    sources: list[str]
    expr: callable
    a: float = 1.0
    b: float = 0.0
    n_fit: int = 0
    raw_r2: float = float("nan")
    fitted_r2: float = float("nan")

    def apply(self, src: np.ndarray) -> np.ndarray:
        return self.a * self.expr(src) + self.b


RELATIONS: dict[str, tuple[list[str], callable]] = {
    "ei":  (["egc", "eea"], lambda d: d[:, 0] + d[:, 1]),
    "eea": (["ei", "egc"],  lambda d: d[:, 0] - d[:, 1]),
    "egb": (["egc"],        lambda d: d[:, 0]),
    "eps": (["nc"],         lambda d: d[:, 0] ** 2),
    "nc":  (["eps"],        lambda d: np.sqrt(np.clip(d[:, 0], 0, None))),
}


def wide_table(train: pd.DataFrame) -> pd.DataFrame:
    """canon -> one column per target_type (NaN where not measured)."""
    return train.pivot_table(index="canon", columns="target_type",
                             values="target", aggfunc="mean")


def _r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    ss = ((y - y.mean()) ** 2).sum()
    return float("nan") if ss == 0 else 1.0 - float(((y - p) ** 2).sum()) / ss


def fit_relations(train: pd.DataFrame, min_n: int = 20) -> dict[str, Relation]:
    """Calibrate every relation on the co-observed rows of `train` ONLY.

    Must be called with the CV training fold, never the full frame, or the
    calibration leaks validation labels.
    """
    w = wide_table(train)
    out: dict[str, Relation] = {}
    for tgt, (srcs, expr) in RELATIONS.items():
        if tgt not in w.columns or any(s not in w.columns for s in srcs):
            continue
        d = w.dropna(subset=[tgt] + srcs)
        if len(d) < min_n:
            continue
        x = expr(d[srcs].values)
        y = d[tgt].values
        if np.std(x) < 1e-12:
            continue
        a, b = np.polyfit(x, y, 1)
        out[tgt] = Relation(tgt, srcs, expr, float(a), float(b), len(d),
                            _r2(y, x), _r2(y, a * x + b))
    return out


def partner_estimate(df: pd.DataFrame, rels: dict[str, Relation],
                     lookup: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Physics estimate per row, using TRUE partner labels from `lookup`.

    Returns (estimate, covered_mask). Rows whose partners are absent get NaN.
    """
    est = np.full(len(df), np.nan)
    for tgt, rel in rels.items():
        m = (df["target_type"] == tgt).values
        if not m.any():
            continue
        idx = df.loc[m, "canon"]
        cols = [c for c in rel.sources if c in lookup.columns]
        if len(cols) != len(rel.sources):
            continue
        src = lookup.reindex(idx)[rel.sources].values
        ok = np.isfinite(src).all(axis=1)
        vals = np.full(len(src), np.nan)
        if ok.any():
            vals[ok] = rel.apply(src[ok])
        est[m] = vals
    return est, np.isfinite(est)


def blend(model_pred: np.ndarray, phys_est: np.ndarray, weight: float) -> np.ndarray:
    """Convex blend where the physics estimate exists, model prediction elsewhere."""
    out = np.asarray(model_pred, float).copy()
    m = np.isfinite(phys_est)
    out[m] = (1.0 - weight) * out[m] + weight * np.asarray(phys_est, float)[m]
    return out


def tune_weight(y_true, model_pred, phys_est, grid=None) -> tuple[float, float]:
    """Pick the blend weight that maximises R2 on the rows physics covers.

    Tune on OOF predictions, per target. Returns (best_weight, best_r2).
    """
    grid = grid if grid is not None else np.linspace(0.0, 1.0, 21)
    m = np.isfinite(phys_est)
    if m.sum() < 10:
        return 0.0, _r2(y_true, model_pred)
    best_w, best = 0.0, _r2(y_true, model_pred)
    for w in grid:
        s = _r2(y_true, blend(model_pred, phys_est, w))
        if s > best:
            best, best_w = s, float(w)
    return best_w, best


def report(rels: dict[str, Relation]) -> None:
    print(f"{'target':<7}{'sources':<14}{'n':>5}{'raw R2':>10}{'fitted R2':>12}{'a':>8}{'b':>8}")
    for t, r in rels.items():
        print(f"{t:<7}{'+'.join(r.sources):<14}{r.n_fit:>5}{r.raw_r2:>10.4f}"
              f"{r.fitted_r2:>12.4f}{r.a:>8.3f}{r.b:>8.3f}")
