"""Multi-task neural network: one shared trunk over the 2978-dim feature matrix,
seven per-property heads.

Why multi-task here
-------------------
The four small properties (eps/nc/ei/eea, ~220 rows each) are 4/7 of the score and
have far too few rows to support an independent deep model. But 5920 distinct
polymers carry *some* label, and the polymers that carry eps also tend to carry
nc/ei/eea/egc -- so a shared trunk trained on the whole long table sees 27x more
molecules than an eea-only model does, and only the 8k-parameter eea head has to
be learned from 221 rows.

Two things make or break this model, and both are handled per fold, fit on the
fold's TRAINING rows only:

1. y is standardised PER TARGET. tg reaches 495 and nc 2.76; a shared MSE on raw
   targets is ~99.9% tg and every other head collapses to its mean.
2. X is signed-log1p compressed then standardised. Raw RDKit descriptors reach
   3.4e38 (Ipc and friends), which is an instant NaN through a BatchNorm.

Rows are collapsed to one row per polymer with a (n_mol, 7) label matrix and a
boolean observation mask, so one forward pass updates every head that polymer has
a label for. The loss is a masked MSE, averaged per target then combined with
fixed weights, so tg's 4139 rows do not drown eea's 221.

Competition compliance: trains from scratch, fixed epoch count with a
deterministic cosine schedule (no wall-clock branching, no early stopping against
the validation fold), no file I/O, no pretrained anything.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

NAME = "mtnn"
DESCRIPTION = "multi-task MLP, shared trunk + 7 per-property heads, masked MSE"

try:  # keep the flattened-notebook copy working standalone
    from src.metric import TARGETS
except Exception:  # pragma: no cover
    TARGETS = ["tg", "egc", "egb", "eps", "nc", "ei", "eea"]

# ---------------------------------------------------------------- hyperparams
HP = dict(
    trunk=(1024, 512, 256, 128),
    head_dim_big=64,        # head width for targets with > SMALL_N training rows
    head_dim_small=32,      # head width for the ~220-row targets
    small_n=1000,
    dropout=0.30,
    head_dropout=0.10,
    epochs=140,
    batch_size=512,
    lr=2.0e-3,
    weight_decay=1.0e-4,
    warmup_frac=0.08,
    final_lr_frac=0.02,
    loss_alpha=0.25,        # per-target loss weight = n_t ** alpha (0 = balanced,
                            # 1 = pooled MSE dominated by tg)
    snapshots=3,            # average predictions from the last k epochs
    snapshot_stride=6,
    clip_z=10.0,            # clip standardised features
    min_std=1e-6,
)


# ------------------------------------------------------------------- network
class _MultiTaskNet(nn.Module):
    def __init__(self, n_in: int, head_dims: list[int], hp: dict):
        super().__init__()
        layers: list[nn.Module] = []
        p = n_in
        for width in hp["trunk"]:
            layers += [nn.Linear(p, width), nn.BatchNorm1d(width), nn.ReLU(),
                       nn.Dropout(hp["dropout"])]
            p = width
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(p, h), nn.ReLU(), nn.Dropout(hp["head_dropout"]),
                          nn.Linear(h, 1))
            for h in head_dims
        ])

    def forward(self, x):
        h = self.trunk(x)
        return torch.cat([head(h) for head in self.heads], dim=1)


# --------------------------------------------------------------- preparation
def _signed_log1p(X: np.ndarray) -> np.ndarray:
    """Tame the dynamic range without fitting anything (so it cannot leak).

    Monotone and sign-preserving: 3.4e38 -> 88.7, and the 2727 binary fingerprint
    bits are merely rescaled 0/1 -> 0/0.693.
    """
    Z = np.asarray(X, dtype=np.float64)
    Z = np.nan_to_num(Z, nan=0.0, posinf=np.finfo(np.float32).max,
                      neginf=-np.finfo(np.float32).max)
    return (np.sign(Z) * np.log1p(np.abs(Z))).astype(np.float32)


def _collapse(df, X, tgt_index: dict[str, int], fold_id=None):
    """Long rows -> one row per polymer, plus (n_mol, 7) label matrix and mask.

    Returns mol_of_row (row -> molecule slot), Xm, Y, M, mol_fold.
    """
    canon = df["canon"].to_numpy()
    slot: dict[str, int] = {}
    mol_of_row = np.empty(len(df), dtype=np.int64)
    first_row: list[int] = []
    for i, c in enumerate(canon):
        j = slot.get(c)
        if j is None:
            j = len(first_row)
            slot[c] = j
            first_row.append(i)
        mol_of_row[i] = j
    first_row_arr = np.asarray(first_row, dtype=np.int64)
    n_mol = len(first_row_arr)

    Xm = np.ascontiguousarray(X[first_row_arr])

    Y = np.zeros((n_mol, len(tgt_index)), dtype=np.float32)
    M = np.zeros((n_mol, len(tgt_index)), dtype=bool)
    if "target" in df.columns:
        tt = df["target_type"].to_numpy()
        yv = df["target"].to_numpy(dtype=np.float64)
        cnt = np.zeros_like(Y)
        for r in range(len(df)):
            c = tgt_index[tt[r]]
            m = mol_of_row[r]
            Y[m, c] += yv[r]
            cnt[m, c] += 1.0
            M[m, c] = True
        Y = np.divide(Y, np.where(cnt > 0, cnt, 1.0)).astype(np.float32)

    # Fold assignment is per (molecule, target) CELL, not per molecule.
    # This project uses per-property folds, so one polymer legitimately has its
    # tg row in fold 3 and its egc row in fold 7. That is not a bug to reject --
    # it is what mirrors inference, where a test polymer's other measured
    # properties are in train and the network has already seen its structure.
    # A molecule therefore trains on its non-held-out targets while being
    # predicted for its held-out ones.
    cell_fold = None
    if fold_id is not None:
        fid = np.asarray(fold_id).astype(np.int64)
        cell_fold = np.full((n_mol, len(tgt_index)), -1, dtype=np.int64)
        tt_arr = df["target_type"].to_numpy()
        for r in range(len(df)):
            cell_fold[mol_of_row[r], tgt_index[tt_arr[r]]] = fid[r]
    return mol_of_row, Xm, Y, M, cell_fold


# ------------------------------------------------------------------ training
def _train_one(Xtr, Ytr, Mtr, head_dims, w_task, hp, seed, device, predict_sets):
    """Train one network and return averaged snapshot predictions for each
    matrix in `predict_sets` (standardised-y space)."""
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dev = torch.device(device)
    net = _MultiTaskNet(Xtr.shape[1], head_dims, hp).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"],
                            weight_decay=hp["weight_decay"])

    xt = torch.from_numpy(Xtr).to(dev)
    yt = torch.from_numpy(Ytr).to(dev)
    mt = torch.from_numpy(Mtr.astype(np.float32)).to(dev)
    wt = torch.from_numpy(w_task.astype(np.float32)).to(dev)

    n = xt.shape[0]
    bs = min(hp["batch_size"], n)
    steps_per_epoch = max(1, int(np.ceil(n / bs)))
    total_steps = hp["epochs"] * steps_per_epoch
    warm = max(1, int(hp["warmup_frac"] * total_steps))
    fl = hp["final_lr_frac"]

    def lr_at(step: int) -> float:
        if step < warm:
            return hp["lr"] * (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return hp["lr"] * (fl + (1.0 - fl) * 0.5 * (1.0 + np.cos(np.pi * prog)))

    rng = np.random.default_rng(seed)
    snap_epochs = {hp["epochs"] - 1 - k * hp["snapshot_stride"]
                   for k in range(hp["snapshots"])}
    snap_epochs = {e for e in snap_epochs if e >= 0}
    acc = [np.zeros((len(P), len(head_dims)), dtype=np.float64) for P in predict_sets]
    n_snap = 0

    step = 0
    for ep in range(hp["epochs"]):
        net.train()
        order = rng.permutation(n)
        for s in range(steps_per_epoch):
            idx = torch.from_numpy(order[s * bs:(s + 1) * bs]).to(dev)
            if idx.numel() < 2:      # BatchNorm needs >= 2 rows
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            step += 1
            xb, yb, mb = xt[idx], yt[idx], mt[idx]
            pred = net(xb)
            se = (pred - yb) ** 2 * mb
            cnt = mb.sum(0)
            per_task = se.sum(0) / cnt.clamp(min=1.0)
            present = (cnt > 0).float()
            w = wt * present
            loss = (w * per_task).sum() / w.sum().clamp(min=1e-8)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

        if ep in snap_epochs:
            net.eval()
            with torch.no_grad():
                for k, P in enumerate(predict_sets):
                    if len(P) == 0:
                        continue
                    pt = torch.from_numpy(P).to(dev)
                    out = []
                    for b in range(0, len(P), 4096):
                        out.append(net(pt[b:b + 4096]).cpu().numpy())
                    acc[k] += np.vstack(out).astype(np.float64)
            n_snap += 1
    return [a / max(1, n_snap) for a in acc]


# -------------------------------------------------------------------- driver
def oof_and_test(train_df, X_tr, test_df, X_te, fold_id, seeds,
                 n_jobs=None, device=None, hp: dict | None = None):
    """Out-of-fold and test predictions from the multi-task network.

    train_df : long format, columns target_type / target / canon
    X_tr     : float32 (len(train_df), n_feat), row-aligned with train_df
    test_df  : columns target_type / canon
    X_te     : float32 (len(test_df), n_feat), row-aligned with test_df
    fold_id  : int array (len(train_df),), grouped by polymer
    seeds    : list[int]; trained once per seed, predictions averaged
    returns  : (oof, test_pred) float64 1-D, len(train_df) / len(test_df)
    """
    hp = {**HP, **(hp or {})}
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if n_jobs:
        torch.set_num_threads(int(n_jobs))
    seeds = list(seeds) if seeds is not None else [42]

    tgt_index = {t: i for i, t in enumerate(TARGETS)}
    n_t = len(TARGETS)
    unknown = (set(train_df["target_type"]) | set(test_df["target_type"])) - set(TARGETS)
    if unknown:
        raise ValueError(f"mtnn: unrecognised target_type {sorted(unknown)}")

    X_tr = np.asarray(X_tr, dtype=np.float32)
    X_te = np.asarray(X_te, dtype=np.float32)
    if X_tr.shape[1] != X_te.shape[1]:
        raise ValueError("mtnn: train/test feature widths differ")

    Ztr_full = _signed_log1p(X_tr)
    Zte_full = _signed_log1p(X_te)

    mol_of_row, Zm, Y, M, cell_fold = _collapse(train_df, Ztr_full, tgt_index, fold_id)
    te_mol_of_row, Zte, _, _, _ = _collapse(test_df, Zte_full, tgt_index)
    te_col = test_df["target_type"].map(tgt_index).to_numpy(dtype=np.int64)
    tr_col = train_df["target_type"].map(tgt_index).to_numpy(dtype=np.int64)

    counts = M.sum(0).astype(np.float64)
    head_dims = [hp["head_dim_big"] if c > hp["small_n"] else hp["head_dim_small"]
                 for c in counts]
    w_task = np.where(counts > 0, np.maximum(counts, 1.0) ** hp["loss_alpha"], 0.0)

    folds = np.unique(cell_fold[cell_fold >= 0])
    oof_mol = np.zeros((len(Zm), n_t), dtype=np.float64)
    oof_hits = np.zeros((len(Zm), n_t), dtype=np.int64)
    test_mol = np.zeros((len(Zte), n_t), dtype=np.float64)
    test_hits = 0

    for f in folds:
        Mva = M & (cell_fold == f)          # cells held out this fold
        Mtr = M & (cell_fold != f)          # cells trainable this fold
        tr = Mtr.any(axis=1)                # molecules with something to learn from
        va = Mva.any(axis=1)                # molecules with something to predict
        if tr.sum() < 10 or va.sum() == 0:
            continue

        # --- feature scaler: fit on this fold's TRAINING molecules only
        mu = Zm[tr].mean(0)
        sd = Zm[tr].std(0)
        keep = sd > hp["min_std"]
        mu_k, sd_k = mu[keep], sd[keep]
        cz = hp["clip_z"]
        Xtr = np.clip((Zm[tr][:, keep] - mu_k) / sd_k, -cz, cz).astype(np.float32)
        Xva = np.clip((Zm[va][:, keep] - mu_k) / sd_k, -cz, cz).astype(np.float32)
        Xte_f = np.clip((Zte[:, keep] - mu_k) / sd_k, -cz, cz).astype(np.float32)

        # --- y scaler: per target, fold-training cells of that target only
        Mtr_sub = Mtr[tr]
        Ytr_raw = Y[tr]
        y_mu = np.zeros(n_t, dtype=np.float64)
        y_sd = np.ones(n_t, dtype=np.float64)
        Ytr = np.zeros_like(Ytr_raw)
        for c in range(n_t):
            m = Mtr_sub[:, c]
            if m.sum() < 2:
                continue
            vals = Ytr_raw[m, c].astype(np.float64)
            y_mu[c] = vals.mean()
            s = vals.std()
            y_sd[c] = s if s > 1e-9 else 1.0
            Ytr[m, c] = ((vals - y_mu[c]) / y_sd[c]).astype(np.float32)

        for seed in seeds:
            pv, pt = _train_one(Xtr, Ytr, Mtr_sub, head_dims, w_task, hp,
                                int(seed) + 1009 * int(f), device, [Xva, Xte_f])
            pv = pv * y_sd + y_mu
            pt = pt * y_sd + y_mu
            # accumulate ONLY the cells actually held out this fold
            oof_mol[va] += pv * Mva[va]
            test_mol += pt
            test_hits += 1
        oof_hits[va] += Mva[va] * len(seeds)

    seen = oof_hits[mol_of_row, tr_col]
    if (seen == 0).any():
        raise ValueError(
            f"mtnn: {int((seen == 0).sum())} training rows were never held out")
    oof_mol = np.divide(oof_mol, np.where(oof_hits > 0, oof_hits, 1))
    test_mol /= max(1, test_hits)

    oof = oof_mol[mol_of_row, tr_col].astype(np.float64)
    test_pred = test_mol[te_mol_of_row, te_col].astype(np.float64)

    if not np.isfinite(oof).all():
        raise ValueError("mtnn: non-finite values in oof")
    if not np.isfinite(test_pred).all():
        raise ValueError("mtnn: non-finite values in test_pred")
    assert oof.shape == (len(train_df),) and test_pred.shape == (len(test_df),)
    return oof, test_pred
