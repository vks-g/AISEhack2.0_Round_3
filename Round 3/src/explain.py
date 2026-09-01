"""Explainability -- the other judged Round 3 theme.

Exact TreeSHAP attributions straight out of LightGBM (`pred_contrib=True`). This
is the same algorithm the `shap` package calls, computed inside LightGBM itself,
so the notebook gains no dependency that could fail to install.

Three views, because "which feature matters" is the wrong granularity for 2,978
mostly-binary fingerprint bits:

  1. per-target top individual features
  2. per-target attribution by FEATURE FAMILY (descriptors vs Morgan bits vs
     MACCS vs polymer terms vs functional groups) -- this is the view that
     actually reads as chemistry
  3. the named RDKit descriptors and SMARTS groups only, which are the ones a
     materials scientist can interpret directly

    ./.venv/bin/python -m src.explain --config lgbm
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.cv import load_config
from src.data import load_train
from src.features import feature_names, featurize
from src.metric import TARGETS

FAMILY_LABEL = {
    "rd": "RDKit descriptors", "mfp2": "Morgan r=2", "mfp3": "Morgan r=3",
    "ap": "atom pair", "tt": "topological torsion", "mac": "MACCS keys",
    "po": "polymer-specific", "grp": "functional groups",
    "true": "partner property", "has": "partner present", "n": "partner count",
}

INTERPRETABLE = ("rd_", "po_", "grp_", "true_", "has_", "n_")


def shap_table(model, X: np.ndarray, columns: list[str]) -> pd.Series:
    """Mean |SHAP| per column. Last column of pred_contrib is the base value."""
    contrib = model.predict(X, pred_contrib=True)
    vals = np.abs(contrib[:, :-1]).mean(axis=0)
    return pd.Series(vals, index=columns).sort_values(ascending=False)


def explain(config_name: str, seed: int = 42, top_k: int = 12,
            max_rows: int = 800) -> dict[str, pd.Series]:
    cfg = load_config(config_name)
    train = load_train()
    state = cfg.fit(train, seed=seed)
    models = state.get("models", {})
    if not models:
        raise ValueError(f"config '{config_name}' exposes no per-target models to explain")

    base_cols = feature_names()
    out = {}
    for t in TARGETS:
        if t not in models:
            continue
        m = (train["target_type"] == t).values
        sub = train.loc[m].reset_index(drop=True)
        if len(sub) > max_rows:
            sub = sub.sample(max_rows, random_state=seed).reset_index(drop=True)
        X = featurize(sub["canon"])

        cols = list(base_cols)
        # Configs that add partner features widen the design matrix; rebuild it
        # through the config's own path so the columns line up.
        if hasattr(cfg, "_design"):
            from src import partners
            (block,), _ = partners.build(state["train"], [sub])
            kept = partners.drop_leaky(block, t)
            X = np.hstack([X, kept.values.astype(np.float32)])
            cols = cols + list(kept.columns)

        model = models[t]
        booster = model.booster_ if hasattr(model, "booster_") else model
        s = shap_table(booster, X, cols)
        out[t] = s

        print(f"\n===== {t}  (n={int(m.sum())}) =====")
        fam = s.groupby([c.split("_", 1)[0] for c in s.index]).sum()
        fam = (fam / fam.sum() * 100).sort_values(ascending=False)
        print("  attribution by family:")
        for k, v in fam.items():
            if v >= 0.5:
                print(f"    {FAMILY_LABEL.get(k, k):<22} {v:5.1f}%")
        print(f"  top {top_k} interpretable features:")
        interp = s[[c for c in s.index if c.startswith(INTERPRETABLE)]]
        for k, v in interp.head(top_k).items():
            print(f"    {k:<34} {v:.4g}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=12)
    a = p.parse_args()
    explain(a.config, seed=a.seed, top_k=a.top_k)
