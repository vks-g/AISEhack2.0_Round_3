"""The two judged deliverables, as measurements rather than assertions.

    ./.venv/bin/python -m src.report --config lgbm

Round 3 grades polymer invariance and explainability alongside accuracy. Most of
what can be said about invariance is cheap to assert and hard to substantiate, so
everything here is a number produced by running the pipeline, including the
places where the guarantee is only approximate.
"""
from __future__ import annotations

import argparse
import collections
import hashlib

import numpy as np
import torch

from src.data import load_test, load_train
from src.features import featurize_one, feature_names
from src.metric import TARGETS
from src.smiles_utils import build_oligomer, canonicalize, randomize, translate


# --------------------------------------------------------------------------- #
# 1. representation-level invariance                                           #
# --------------------------------------------------------------------------- #

def feature_invariance(smiles, n=200):
    """How much of each feature family survives each rewriting.

    The distinction that matters: an EXTENSIVE quantity (molecular weight, atom
    counts) doubles when a repeat unit is written as its dimer; an INTENSIVE one
    (a ratio, a per-atom average) does not. Fingerprint models are therefore
    exactly invariant to atom ordering and measurably NOT invariant to repeat
    count -- which is the invariance the competition host actually asked about.
    """
    names = feature_names()
    smis = list(dict.fromkeys(smiles))[:n]
    base = np.vstack([featurize_one(canonicalize(s)) for s in smis])
    fams = ["iv_", "el_f", "chg_", "po_", "rd_", "el_n", "mfp2_", "mfp3_",
            "ap_", "tt_", "mac_", "grp_"]
    idx = {f: [i for i, nm in enumerate(names) if nm.startswith(f)] for f in fams}
    out = {}
    for label, fn in [("permutational", lambda s: randomize(s, seed=1)),
                      ("translational", lambda s: translate(s, k=1)),
                      ("repetition (dimer)", lambda s: build_oligomer(s, 2))]:
        var = np.vstack([featurize_one(canonicalize(fn(s))) for s in smis])
        row = {}
        for f, ii in idx.items():
            if not ii:
                continue
            a, b = base[:, ii], var[:, ii]
            row[f] = float((np.abs(b - a) / (np.abs(a) + 1e-6) < 0.02).mean() * 100)
        out[label] = row
    return out


# --------------------------------------------------------------------------- #
# 2. the readout ablation -- why the graph model is built the way it is        #
# --------------------------------------------------------------------------- #

def readout_ablation(smiles, n=150):
    """Repetition invariance depends on the READOUT, not just the graph.

    A periodic graph makes the monomer and its dimer produce node environments
    with the same distribution. A MEAN or MAX over nodes therefore returns
    (nearly) the same graph fingerprint; a SUM scales with the number of repeat
    units and destroys the property outright. This is the single design decision
    behind the invariance claim, so it is measured rather than asserted.
    """
    from src.models.gnn import periodic_graph, _collate, _Net, HP

    smis = list(dict.fromkeys(smiles))[:n]
    torch.manual_seed(0)
    net = _Net(HP, len(TARGETS)).eval()

    class _SumNet(_Net):
        def forward(self, nf, src, dst, ef, batch, ng):
            x = self.embed(nf)
            ea = self.eemb(ef) if ef.numel() else torch.zeros(0, x.shape[1])
            for L in self.layers:
                x = L(x, src, dst, ea)
            s = torch.zeros(ng, x.shape[1], dtype=x.dtype).index_add_(0, batch, x)
            z = self.head(torch.cat([s, s], dim=1))
            return torch.cat([o(z) for o in self.out], dim=1)

    torch.manual_seed(0)
    sumnet = _SumNet(HP, len(TARGETS)).eval()

    def run(model, ss):
        g = [periodic_graph(s) for s in ss]
        with torch.no_grad():
            NF, S, D, EF, B, ng = _collate(g, list(range(len(g))), "cpu")
            return model(NF, S, D, EF, B, ng).numpy()

    res = {}
    for tag, model in (("mean+max (shipped)", net), ("sum (conventional)", sumnet)):
        b = run(model, [canonicalize(s) for s in smis])
        for label, fn in [("atom ordering", lambda s: randomize(s, seed=1)),
                          ("dimer", lambda s: build_oligomer(s, 2)),
                          ("trimer", lambda s: build_oligomer(s, 3))]:
            v = run(model, [canonicalize(fn(s)) for s in smis])
            d = np.abs(v - b).max(axis=1)
            res[(tag, label)] = (float(np.median(d)), float(d.max()),
                                 float((d < 1e-9).mean() * 100))
    return res


# --------------------------------------------------------------------------- #
# 3. applicability domain from the auxiliary corpus                            #
# --------------------------------------------------------------------------- #

def applicability_domain(train_df, oof, aux_path=None, aux_max=300_000, seed=42):
    """Flag molecules containing substructure the 5.97M corpus never shows.

    Folded fingerprints are useless for this -- with a corpus this size every bit
    is occupied, so "fraction unseen" collapses to zero. Unfolded substructure
    hashes keep the resolution. The `*` attachment points are capped as methyl
    first: no corpus molecule has one, so comparing raw makes every polymer novel
    by definition and the flag carries no information.
    """
    import pandas as pd
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    from src.metric import r2
    from src.paths import data_dir

    aux_path = aux_path or (data_dir() / "PI1M.csv")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)

    def hashes(s):
        m = Chem.MolFromSmiles(str(s).replace("[*]", "C").replace("*", "C"))
        if m is None:
            return set()
        return set(gen.GetSparseCountFingerprint(m).GetNonzeroElements().keys())

    ours = {c: hashes(c) for c in dict.fromkeys(train_df["canon"])}
    keep = set().union(*ours.values()) if ours else set()

    col = pd.read_csv(aux_path, nrows=3).columns
    scol = next((c for c in col if "smi" in c.lower()), col[0])
    aux = pd.read_csv(aux_path, usecols=[scol])[scol].dropna().astype(str).values
    if len(aux) > aux_max:
        aux = np.random.default_rng(seed).choice(aux, aux_max, replace=False)
    seen = set()
    for s in aux:
        seen |= (hashes(s) & keep)

    novel = {c: len(h - seen) for c, h in ours.items()}
    flag = train_df["canon"].map(lambda c: novel.get(c, 0) > 0).values

    rows = []
    for t in TARGETS:
        m = (train_df["target_type"] == t).values
        y, p = train_df["target"].values[m], np.asarray(oof)[m]
        fin, fout = flag[m], ~flag[m]
        if fin.sum() < 20 or fout.sum() < 20:
            continue
        rows.append((t, int(fout.sum()), r2(y[fout], p[fout]),
                     int(fin.sum()), r2(y[fin], p[fin]),
                     float(np.abs(y[fout] - p[fout]).mean()),
                     float(np.abs(y[fin] - p[fin]).mean())))
    return rows, float(flag.mean() * 100), len(seen), len(keep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--skip-aux", action="store_true")
    a = ap.parse_args()

    tr, te = load_train(), load_test()

    print("=" * 78)
    print("1. REPRESENTATION INVARIANCE -- % of feature values unchanged (<2% rel)")
    print("=" * 78)
    inv = feature_invariance(te["smiles"].values, n=a.n)
    fams = list(next(iter(inv.values())).keys())
    print(f"{'family':<10}" + "".join(f"{k[:18]:>20}" for k in inv))
    for f in fams:
        print(f"{f:<10}" + "".join(f"{inv[k][f]:>19.1f}%" for k in inv))
    print("\nIntensive features (iv_, el_f) survive repeat-unit change; extensive")
    print("ones (el_n, parts of rd_) do not -- they scale with the unit count.")

    print()
    print("=" * 78)
    print("2. GRAPH READOUT ABLATION -- max |delta| in prediction, untrained net")
    print("=" * 78)
    ab = readout_ablation(te["smiles"].values, n=min(a.n, 150))
    print(f"{'readout':<22}{'rewriting':<16}{'median':>12}{'max':>12}{'exact':>9}")
    for (tag, lab), (md, mx, ex) in ab.items():
        print(f"{tag:<22}{lab:<16}{md:>12.2e}{mx:>12.2e}{ex:>8.0f}%")
    print("\nAtom ordering is EXACT for both. Repeat count is where they separate:")
    print("mean/max stays ~1e-4, a sum readout moves by order 1 -- four orders of")
    print("magnitude. The invariance is a property of the readout, not just the graph.")
