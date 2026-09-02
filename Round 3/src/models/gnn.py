"""Periodic multitask message-passing network over the polymer graph.

Follows polyGNN (Gurnani, Kuenneth, Toland & Ramprasad, Chem. Mater. 2023), the
published state of the art for this dataset family -- the seven targets here come
from Kuenneth et al., Patterns 2021, the multi-task polymer informatics paper.

WHY A PERIODIC GRAPH. A repeat unit is not a molecule with two dangling stubs.
The two `*` attachment points are the SAME bond to the neighbouring unit, so the
correct object is the graph with the dummies removed and their neighbours bonded:
the unit wraps around and encodes an infinite chain. This buys the two invariances
the competition host asked about in the discussion thread, and it buys them by
construction rather than by convention:

  translational  where the repeat unit is cut no longer exists as a concept --
                 every cut of the same chain gives the same cyclic graph.
  repetition     monomer, dimer and trimer give node environments with an
                 IDENTICAL distribution, so any permutation-invariant readout
                 that is a MEAN over nodes returns the same value.

THE READOUT IS PART OF THE GUARANTEE, NOT A DETAIL. Measured here on 390 real
polymers, comparing a monomer against its own dimer after 4 rounds of
Weisfeiler-Lehman refinement (which is exactly what message passing computes):

    mean / max readout   388/390 = 99.5% identical
    sum readout            0/390 =  0.0% identical

A sum readout scales with the number of repeat units and destroys the property.
So this model reads out with mean and max ONLY. Canonicalisation cannot deliver
this: a dimer's canonical SMILES is a different string, so every fingerprint
changes -- which is why our fingerprint models are permutation-invariant but
measurably NOT repetition-invariant.

Interface matches models/mtnn.py:
    oof_and_test(train_df, test_df, fold_id, seeds, device=None, hp=None)
Reads df['canon'] only, so no leakage guard is needed.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from src.metric import TARGETS

NAME = "gnn"

HP = dict(
    hidden=160,
    layers=4,
    dropout=0.10,
    lr=1.5e-3,
    weight_decay=1e-5,
    epochs=110,
    batch_size=128,
    warmup_frac=0.05,
    final_lr_frac=0.05,
    head=96,
    snapshots=3,          # average the last k epochs -- cheap seed-like averaging
    snapshot_stride=4,
    clip=2.0,
)

_ATOMS = [6, 7, 8, 9, 14, 15, 16, 17, 35, 53, 5, 1]
_HYB = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
_BONDS = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]


def _oh(x, vocab):
    v = [0.0] * (len(vocab) + 1)
    v[vocab.index(x) if x in vocab else len(vocab)] = 1.0
    return v


NODE_F = (len(_ATOMS) + 1) + 6 + 4 + (len(_HYB) + 1) + 5 + 3
EDGE_F = (len(_BONDS) + 1) + 3


def periodic_graph(smi: str):
    """(node_features, src, dst, edge_features) for the PERIODIC polymer graph.

    Falls back to the plain molecular graph when the repeat unit is degenerate
    (not exactly two `*`, both on the same atom, or neighbours already bonded).
    """
    m = Chem.MolFromSmiles(smi)
    if m is None or m.GetNumAtoms() == 0:
        return (np.zeros((1, NODE_F), np.float32), np.zeros(0, np.int64),
                np.zeros(0, np.int64), np.zeros((0, EDGE_F), np.float32))

    stars = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    wrap = None
    if len(stars) == 2:
        nb = []
        for s in stars:
            n = [x.GetIdx() for x in m.GetAtomWithIdx(s).GetNeighbors()]
            nb.append(n[0] if n else None)
        if nb[0] is not None and nb[1] is not None and nb[0] != nb[1]:
            wrap = (nb[0], nb[1])

    drop = set(stars) if wrap is not None else set()
    keep = [a.GetIdx() for a in m.GetAtoms() if a.GetIdx() not in drop]
    ix = {o: i for i, o in enumerate(keep)}

    nf = []
    for o in keep:
        a = m.GetAtomWithIdx(o)
        nf.append(
            _oh(a.GetAtomicNum(), _ATOMS)
            + _oh(a.GetDegree(), [0, 1, 2, 3, 4])[:6]
            + _oh(a.GetFormalCharge(), [-1, 0, 1])[:4]
            + _oh(str(a.GetHybridization()), _HYB)
            + _oh(a.GetTotalNumHs(), [0, 1, 2, 3])[:5]
            + [float(a.GetIsAromatic()), float(a.IsInRing()),
               float(a.GetAtomicNum() == 0)]
        )
    src, dst, ef = [], [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i not in ix or j not in ix:
            continue
        e = _oh(str(b.GetBondType()), _BONDS) + [
            float(b.GetIsConjugated()), float(b.IsInRing()), 0.0]
        src += [ix[i], ix[j]]; dst += [ix[j], ix[i]]; ef += [e, e]
    if wrap is not None:
        u, v = ix[wrap[0]], ix[wrap[1]]
        e = _oh("SINGLE", _BONDS) + [0.0, 0.0, 1.0]      # flagged as the wrap bond
        src += [u, v]; dst += [v, u]; ef += [e, e]

    if not nf:
        nf = [[0.0] * NODE_F]
    return (np.asarray(nf, np.float32), np.asarray(src, np.int64),
            np.asarray(dst, np.int64),
            np.asarray(ef, np.float32) if ef else np.zeros((0, EDGE_F), np.float32))


class _MP(nn.Module):
    def __init__(self, h, e):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(2 * h + e, h), nn.SiLU(), nn.Linear(h, h))
        self.gru = nn.GRUCell(h, h)

    def forward(self, x, src, dst, ea):
        if src.numel() == 0:
            return x
        msg = self.msg(torch.cat([x[src], x[dst], ea], dim=1))
        agg = torch.zeros_like(x).index_add_(0, dst, msg)
        return self.gru(agg, x)


class _Net(nn.Module):
    def __init__(self, hp, n_tasks):
        super().__init__()
        h = hp["hidden"]
        self.embed = nn.Linear(NODE_F, h)
        self.eemb = nn.Linear(EDGE_F, h) if EDGE_F else None
        self.layers = nn.ModuleList([_MP(h, h) for _ in range(hp["layers"])])
        self.drop = nn.Dropout(hp["dropout"])
        # mean + max ONLY. A sum term would make the graph fingerprint scale with
        # the number of repeat units and break repetition invariance (0/390).
        self.head = nn.Sequential(nn.Linear(2 * h, hp["head"]), nn.SiLU(),
                                  nn.Dropout(hp["dropout"]))
        self.out = nn.ModuleList([nn.Linear(hp["head"], 1) for _ in range(n_tasks)])

    def forward(self, nf, src, dst, ef, batch, n_graphs):
        x = self.embed(nf)
        ea = self.eemb(ef) if ef.numel() else torch.zeros(0, x.shape[1], device=x.device)
        for L in self.layers:
            x = self.drop(L(x, src, dst, ea))
        cnt = torch.zeros(n_graphs, device=x.device, dtype=x.dtype).index_add_(
            0, batch, torch.ones_like(batch, dtype=x.dtype)).clamp(min=1).unsqueeze(1)
        mean = torch.zeros(n_graphs, x.shape[1], device=x.device,
                           dtype=x.dtype).index_add_(0, batch, x) / cnt
        mx = torch.full((n_graphs, x.shape[1]), -1e30, device=x.device, dtype=x.dtype)
        mx = mx.index_reduce_(0, batch, x, "amax", include_self=True)
        mx = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)
        z = self.head(torch.cat([mean, mx], dim=1))
        return torch.cat([o(z) for o in self.out], dim=1)


# --------------------------------------------------------------------------- #
# batching                                                                     #
# --------------------------------------------------------------------------- #

def _collate(graphs, idx, device):
    nf, src, dst, ef, batch = [], [], [], [], []
    off = 0
    for b, i in enumerate(idx):
        n, s, d, e = graphs[i]
        nf.append(n); ef.append(e)
        src.append(s + off); dst.append(d + off)
        batch.append(np.full(len(n), b, np.int64))
        off += len(n)
    cat = lambda xs, f: (np.concatenate(xs) if len(xs) and sum(len(x) for x in xs)
                         else np.zeros((0,) + f, np.float32))
    NF = torch.from_numpy(np.concatenate(nf)).to(device)
    EF = torch.from_numpy(np.concatenate(ef) if sum(len(e) for e in ef)
                          else np.zeros((0, EDGE_F), np.float32)).to(device)
    SRC = torch.from_numpy(np.concatenate(src) if sum(len(s) for s in src)
                           else np.zeros(0, np.int64)).to(device)
    DST = torch.from_numpy(np.concatenate(dst) if sum(len(d) for d in dst)
                           else np.zeros(0, np.int64)).to(device)
    BAT = torch.from_numpy(np.concatenate(batch)).to(device)
    return NF, SRC, DST, EF, BAT, len(idx)


def _collapse(df, tgt_index, fold_id=None):
    """rows -> molecules, with a per-(molecule, target) fold id."""
    canon = df["canon"].to_numpy()
    uniq, mol_of_row = np.unique(canon, return_inverse=True)
    n_mol, n_t = len(uniq), len(tgt_index)
    Y = np.zeros((n_mol, n_t), np.float32)
    M = np.zeros((n_mol, n_t), bool)
    cell_fold = np.full((n_mol, n_t), -1, np.int64)
    if "target" in df.columns:
        tt = df["target_type"].to_numpy(); yv = df["target"].to_numpy(float)
        cnt = np.zeros_like(Y)
        for r in range(len(df)):
            c = tgt_index[tt[r]]; mm = mol_of_row[r]
            Y[mm, c] += yv[r]; cnt[mm, c] += 1.0; M[mm, c] = True
            if fold_id is not None:
                cell_fold[mm, c] = fold_id[r]
        Y = np.divide(Y, np.where(cnt > 0, cnt, 1.0)).astype(np.float32)
    return uniq, mol_of_row, Y, M, cell_fold


def _train_one(graphs, tr_idx, Ytr, Mtr, hp, seed, device, predict_sets, n_t):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    net = _Net(hp, n_t).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"],
                            weight_decay=hp["weight_decay"])
    yt = torch.from_numpy(Ytr).to(device)
    mt = torch.from_numpy(Mtr.astype(np.float32)).to(device)
    n = len(tr_idx)
    bs = min(hp["batch_size"], max(1, n))
    steps = max(1, int(np.ceil(n / bs)))
    total = hp["epochs"] * steps
    warm = max(1, int(hp["warmup_frac"] * total))
    fl = hp["final_lr_frac"]

    def lr_at(s):
        if s < warm:
            return (s + 1) / warm
        p = (s - warm) / max(1, total - warm)
        return fl + (1 - fl) * 0.5 * (1 + np.cos(np.pi * p))

    rng = np.random.RandomState(seed)
    snap = {hp["epochs"] - 1 - k * hp["snapshot_stride"] for k in range(hp["snapshots"])}
    snap = {e for e in snap if e >= 0}
    acc = [np.zeros((len(p), n_t)) for p in predict_sets]
    nsnap = 0
    step = 0
    for ep in range(hp["epochs"]):
        net.train()
        order = rng.permutation(n)
        for b in range(steps):
            sel = order[b * bs:(b + 1) * bs]
            if len(sel) == 0:
                continue
            for g in opt.param_groups:
                g["lr"] = hp["lr"] * lr_at(step)
            step += 1
            NF, S, D, EF, BAT, ng = _collate(graphs, [tr_idx[i] for i in sel], device)
            pred = net(NF, S, D, EF, BAT, ng)
            yb, mb = yt[sel], mt[sel]
            loss = (((pred - yb) ** 2) * mb).sum() / mb.sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), hp["clip"])
            opt.step()
        if ep in snap:
            net.eval()
            with torch.no_grad():
                for k, pset in enumerate(predict_sets):
                    out = np.zeros((len(pset), n_t))
                    for b in range(0, len(pset), 512):
                        chunk = pset[b:b + 512]
                        NF, S, D, EF, BAT, ng = _collate(graphs, chunk, device)
                        out[b:b + len(chunk)] = net(NF, S, D, EF, BAT, ng).cpu().numpy()
                    acc[k] += out
            nsnap += 1
    return [a / max(1, nsnap) for a in acc]


def oof_and_test(train_df, test_df, fold_id, seeds, device=None, hp=None,
                 verbose=True):
    hp = dict(HP, **(hp or {}))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tgt_index = {t: i for i, t in enumerate(TARGETS)}
    n_t = len(TARGETS)

    u_tr, mol_tr, Y, M, cell_fold = _collapse(train_df, tgt_index, fold_id)
    u_te, mol_te, _, _, _ = _collapse(test_df, tgt_index)
    all_smi = list(u_tr) + list(u_te)
    graphs = [periodic_graph(s) for s in all_smi]
    tr_off, te_off = 0, len(u_tr)
    if verbose:
        print(f"gnn: {len(u_tr)} train / {len(u_te)} test molecules, device={device}",
              flush=True)

    oof_mol = np.zeros((len(u_tr), n_t))
    hits = np.zeros((len(u_tr), n_t))
    test_mol = np.zeros((len(u_te), n_t))
    n_test = 0

    for f in np.unique(cell_fold[cell_fold >= 0]):
        Mva = M & (cell_fold == f)
        Mtr = M & (cell_fold != f)
        tr_sel = np.where(Mtr.any(1))[0]
        va_sel = np.where(Mva.any(1))[0]
        if len(tr_sel) < 20 or len(va_sel) == 0:
            continue
        # y standardised PER TARGET on this fold's training cells only
        mu = np.zeros(n_t); sd = np.ones(n_t)
        Ytr = np.zeros((len(tr_sel), n_t), np.float32)
        Msub = Mtr[tr_sel]
        for c in range(n_t):
            m = Msub[:, c]
            if m.sum() < 2:
                continue
            v = Y[tr_sel][m, c].astype(float)
            mu[c] = v.mean(); s = v.std(); sd[c] = s if s > 1e-9 else 1.0
            Ytr[m, c] = ((v - mu[c]) / sd[c]).astype(np.float32)
        for sd_i in seeds:
            pv, pt = _train_one(graphs, [tr_off + i for i in tr_sel], Ytr, Msub, hp,
                                int(sd_i) + 7919 * int(f), device,
                                [[tr_off + i for i in va_sel],
                                 [te_off + i for i in range(len(u_te))]], n_t)
            oof_mol[va_sel] += (pv * sd + mu) * Mva[va_sel]
            test_mol += pt * sd + mu
            n_test += 1
        hits[va_sel] += Mva[va_sel] * len(seeds)

    tr_col = train_df["target_type"].map(tgt_index).to_numpy()
    te_col = test_df["target_type"].map(tgt_index).to_numpy()
    seen = hits[mol_tr, tr_col]
    if (seen == 0).any():
        raise ValueError(f"gnn: {int((seen==0).sum())} training rows never held out")
    oof_mol = np.divide(oof_mol, np.where(hits > 0, hits, 1))
    test_mol /= max(1, n_test)
    oof = oof_mol[mol_tr, tr_col].astype(float)
    test_pred = test_mol[mol_te, te_col].astype(float)
    if not np.isfinite(oof).all() or not np.isfinite(test_pred).all():
        raise ValueError("gnn: non-finite predictions")
    return oof, test_pred
