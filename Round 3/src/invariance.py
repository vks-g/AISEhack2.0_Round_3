"""Polymer-invariance audit -- one of the two judged Round 3 themes.

The claim to substantiate: the pipeline returns the SAME prediction for any valid
re-writing of the same polymer. Three rewritings are probed separately, because
they are different guarantees with different strengths:

  permutational  atom ordering        EXACT by construction. Every feature is
                                      computed from the canonical SMILES, and
                                      canonicalisation is idempotent, so the
                                      prediction is bit-identical.
  translational  different cut point  NOT exact. A different cut changes the
                                      graph RDKit sees (the two `*` move), so
                                      descriptors and fingerprints shift.
                                      This measures how much.
  repetition     monomer/dimer/trimer NOT exact, same reason -- an n-mer has n
                                      times the atoms.

MEASURED CONTEXT: train+test contain no genuine oligomer duplicates and only 7
borderline translational groups out of ~9,000 polymers, so this audit is a
robustness certificate for the rubric, not a source of leaderboard points. Do
not spend a submission slot chasing it.

    ./.venv/bin/python -m src.invariance --config lgbm --n 200
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.cv import load_config
from src.data import load_train
from src.metric import TARGETS
from src.smiles_utils import canonicalize, build_oligomer, randomize, translate

KINDS = {
    "permutational": lambda s, i: randomize(s, seed=i),
    "translational": lambda s, i: translate(s, k=i + 1),
    "repetition":    lambda s, i: build_oligomer(s, n=i + 2),
}


def audit(config_name: str, n: int = 200, seed: int = 42, k: int = 2) -> pd.DataFrame:
    cfg = load_config(config_name)
    train = load_train()
    state = cfg.fit(train, seed=seed)

    sample = train.sample(n=min(n, len(train)), random_state=seed).reset_index(drop=True)
    base = cfg.predict(state, sample)

    rows = []
    for kind, make in KINDS.items():
        for i in range(k):
            variant = sample.copy()
            variant["smiles"] = [make(s, i) for s in sample["smiles"]]
            variant["canon"] = variant["smiles"].map(canonicalize)
            # "rewritten" = the string actually changed. For permutational the
            # canonical form must NOT change -- that is the guarantee being tested.
            rewritten = (variant["smiles"] != sample["smiles"]).values
            canon_changed = (variant["canon"] != sample["canon"]).values
            p = cfg.predict(state, variant)
            for j in range(len(sample)):
                rows.append({
                    "kind": kind, "target_type": sample.at[j, "target_type"],
                    "delta": abs(float(p[j] - base[j])),
                    "rewritten": bool(rewritten[j]),
                    "canon_changed": bool(canon_changed[j]),
                    "scale": max(abs(float(sample.at[j, "target"])), 1e-9),
                })
    res = pd.DataFrame(rows)

    def fmt(v):
        if not np.isfinite(v):
            return "     inf"
        return f"{v:8.4f}" if abs(v) < 1e5 else f"{v:8.1e}"

    print(f"\nInvariance audit -- {config_name}, {len(sample)} polymers, "
          f"{k} variants per kind\n")
    print(f"{'kind':<16}{'rewritten':>11}{'canon moved':>13}"
          f"{'median|d|':>12}{'mean|d|':>12}{'max|d|':>12}{'med rel':>10}")
    for kind in KINDS:
        m = res[res.kind == kind]
        eff = m[m.rewritten]
        if not len(eff):
            print(f"{kind:<16}{0:>6}/{len(m):<4}{'-':>13}{'-':>12}{'-':>12}{'-':>12}{'-':>10}")
            continue
        rel = 100 * (eff.delta / eff.scale).median()
        print(f"{kind:<16}{len(eff):>6}/{len(m):<4}{int(eff.canon_changed.sum()):>13}"
              f"{fmt(eff.delta.median())}{fmt(eff.delta.mean())}{fmt(eff.delta.max())}"
              f"{rel:>9.2f}%")

    print(f"\n{'':<10}median |delta| per target, on rewritten variants")
    print(f"{'target':<10}" + "".join(f"{k:>16}" for k in KINDS))
    for t in TARGETS:
        line = f"{t:<10}"
        for kind in KINDS:
            m = res[(res.kind == kind) & (res.target_type == t) & res.rewritten]
            line += f"{fmt(m.delta.median()) if len(m) else '-':>16}"
        print(line)

    perm = res[(res.kind == "permutational") & res.rewritten]
    if not len(perm):
        print("\nWARN: no permutational rewriting actually changed the string; "
              "the exactness claim was not exercised.")
    elif perm.delta.max() > 1e-9:
        print(f"\nFAIL: permutational invariance is not exact "
              f"(max delta {perm.delta.max():.3e}). Something in the pipeline is "
              f"reading `smiles` instead of `canon`.")
    else:
        print(f"\nPASS: permutational invariance is EXACT -- {len(perm)} rewritten "
              f"strings, {int(perm.canon_changed.sum())} of which changed the "
              f"canonical form, max |delta| = 0. Every feature derives from the "
              f"canonical SMILES.")

    rep = res[(res.kind == "repetition") & res.rewritten]
    if len(rep) and rep.delta.median() > 1e3:
        print("NOTE: repetition deltas are enormous. A linear model extrapolates "
              "on size-dependent descriptors (a trimer has 3x the atoms), so this "
              "is a property of the model, not a bug. Compare against a tree model.")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variants", type=int, default=2)
    a = p.parse_args()
    audit(a.config, n=a.n, seed=a.seed, k=a.variants)
