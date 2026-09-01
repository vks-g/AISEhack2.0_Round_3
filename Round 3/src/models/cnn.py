"""Character-level 1-D CNN over the CANONICAL SMILES string, trained from scratch.

Why this model exists
---------------------
Every other model in the stack sees a 2978-dim bag of substructure counts. This
one sees the *string*: character order, branch/ring syntax, local motifs. It is
weaker on its own (mean OOF R2 ~0.60 vs 0.88 for lgbm_physics) but its errors are
decorrelated from the descriptor models, which is the whole point of putting it
in the stack.

Design decisions that are load-bearing
--------------------------------------
* Tokenises ``df['canon']`` (RDKit canonical SMILES), never ``df['smiles']``.
  Feeding raw SMILES would make the prediction depend on how a polymer happened
  to be written and break the pipeline's invariance guarantee.
* Multi-task: one trunk, seven heads, masked loss, y standardised PER TARGET
  using only the fold's training rows. Without the per-target standardisation
  ``tg`` (hundreds) swamps ``nc`` (~1.6) and the four ~220-row targets learn
  nothing. The loss additionally weights each target equally (per-target mean,
  then mean over targets) to match the unweighted-mean-R2 metric -- otherwise
  ``tg``'s 4139 rows own 56% of the gradient.
* The multi-kernel same-padded convolution is implemented as unfold + GEMM
  instead of ``nn.Conv1d``. Numerically identical, but MEASURED on this box
  ``nn.Conv1d`` runs at 13 GFLOPS while ``matmul`` runs at 959; the rewrite took
  a 10-fold epoch from 12.5s to 3.3s. See ``_MultiKernelConv``.
* PAD is token 0 with ``padding_idx=0`` (embedding pinned to the zero vector,
  never updated) and conv outputs at pad positions are masked out before the
  global max-pool. Consequence: a molecule's prediction is bit-identical no
  matter which length-bucket batch it lands in. Without this, length-bucketed
  batching would make predictions depend on batch composition.
* Fixed epoch count, cosine schedule, no early stopping, no wall-clock
  branching. Deterministic given ``seeds``.
* Self-contained: no file I/O, no caching, no module-level state that depends on
  anything not passed in. Safe to flatten into one Kaggle cell.

Competition-rule compliance: trains from scratch every run, no pretrained
weights, no downloads, and nothing deserialised from disk.
"""
from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "cnn"
DESCRIPTION = "character-level SMILES CNN, multi-task, 7 heads, masked loss"

# The 7 targets, lowercase, in the canonical order used everywhere in src/.
# Duplicated rather than imported so the flattened notebook cell stands alone.
TARGETS: list[str] = ["tg", "egc", "egb", "eps", "nc", "ei", "eea"]

# --------------------------------------------------------------------------- #
# hyperparameters -- one dict so the driver can override without editing code   #
# --------------------------------------------------------------------------- #
# MAX_LEN=150 measured on train+test canonical SMILES (10,053 unique strings):
#   mean length 48.6, median 38, p99 147.6.
#   0.92% of rows are truncated; 0.47% of all characters are dropped.
# Cost is driven by the *batch* max length (see length bucketing), not MAX_LEN,
# so raising it buys nothing and lowering it costs accuracy on the long tail.
HP = dict(
    max_len=150,
    emb_dim=64,
    n_filters=96,
    kernels=(3, 5, 7, 11),
    fc_dim=256,
    dropout=0.3,
    epochs=32,
    batch_size=256,
    lr=2e-3,
    weight_decay=1e-5,
    warmup_epochs=3,
    lr_final_frac=0.02,
    grad_clip=5.0,
    pool_batches=8,      # length-bucket pool = pool_batches * batch_size
    clip_to_train_range=True,
)

PAD, UNK = 0, 1


# --------------------------------------------------------------------------- #
# tokenisation                                                                  #
# --------------------------------------------------------------------------- #

def build_vocab(smiles) -> dict[str, int]:
    """Character vocabulary from the TRAINING strings only. ids 0=PAD, 1=UNK."""
    chars = sorted({c for s in smiles for c in str(s)})
    return {c: i + 2 for i, c in enumerate(chars)}


def encode(smiles, vocab: dict[str, int], max_len: int):
    """-> (int64 (n, max_len) left-aligned, zero-padded; int64 (n,) lengths)."""
    n = len(smiles)
    X = np.zeros((n, max_len), dtype=np.int64)
    lens = np.zeros(n, dtype=np.int64)
    for i, s in enumerate(smiles):
        s = str(s)[:max_len]
        if not s:                      # never happens on this data; keep len>=1
            s = "*"
        lens[i] = len(s)
        for j, c in enumerate(s):
            X[i, j] = vocab.get(c, UNK)
    return X, lens


def truncation_stats(smiles, max_len: int) -> dict:
    L = np.array([len(str(s)) for s in smiles])
    return {
        "n": int(L.size), "mean_len": float(L.mean()), "p99": float(np.percentile(L, 99)),
        "frac_rows_truncated": float((L > max_len).mean()),
        "frac_chars_dropped": float(np.clip(L - max_len, 0, None).sum() / L.sum()),
    }


# --------------------------------------------------------------------------- #
# model                                                                         #
# --------------------------------------------------------------------------- #

class _MultiKernelConv(nn.Module):
    """Parallel same-padded Conv1d for several kernel sizes, as unfold + GEMM.

    Exactly equivalent to ``[nn.Conv1d(emb, filt, k, padding=k//2) for k in ks]``
    (odd k only, so 'same' padding is symmetric), but expressed as one matmul per
    kernel so it runs on the BLAS path instead of torch's slow CPU conv1d.
    """

    def __init__(self, emb: int, filt: int, kernels):
        super().__init__()
        self.kernels = tuple(kernels)
        assert all(k % 2 == 1 for k in self.kernels), "kernels must be odd"
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        for k in self.kernels:
            w = torch.empty(emb * k, filt)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))     # nn.Conv1d's own init
            bound = 1.0 / math.sqrt(emb * k)
            self.W.append(nn.Parameter(w))
            self.b.append(nn.Parameter(torch.empty(filt).uniform_(-bound, bound)))
        self.out_dim = filt * len(self.kernels)

    def forward(self, h: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """h (B,L,emb) -> (B, filt*len(kernels)) after masked global max-pool."""
        B, L, E = h.shape
        hp = h.transpose(1, 2)                                   # B,emb,L
        dead = (~valid).unsqueeze(-1)                            # B,L,1
        outs = []
        for k, W, b in zip(self.kernels, self.W, self.b):
            p = k // 2
            u = F.pad(hp, (p, p)).unfold(2, k, 1)                # B,emb,L,k
            u = u.permute(0, 2, 1, 3).reshape(B, L, E * k)       # B,L,emb*k
            z = torch.relu(torch.addmm(b, u.reshape(B * L, E * k), W).view(B, L, -1))
            # relu output is >= 0 and every row has >= 1 valid position, so
            # zeroing the pad positions leaves the max over valid ones intact.
            outs.append(z.masked_fill(dead, 0.0).amax(1))
        return torch.cat(outs, dim=1)


class SmilesCNN(nn.Module):
    def __init__(self, vocab_size: int, n_targets: int, hp: dict):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hp["emb_dim"], padding_idx=PAD)
        nn.init.normal_(self.emb.weight, 0.0, 0.1)
        with torch.no_grad():
            self.emb.weight[PAD].zero_()
        self.conv = _MultiKernelConv(hp["emb_dim"], hp["n_filters"], hp["kernels"])
        self.drop = nn.Dropout(hp["dropout"])
        self.fc = nn.Linear(self.conv.out_dim, hp["fc_dim"])
        self.heads = nn.Linear(hp["fc_dim"], n_targets)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        z = self.conv(h, valid)
        z = self.drop(z)
        z = torch.relu(self.fc(z))
        return self.heads(self.drop(z))


# --------------------------------------------------------------------------- #
# batching                                                                      #
# --------------------------------------------------------------------------- #

def _length_bucketed_batches(lens: np.ndarray, bs: int, pool_batches: int,
                             rng: np.random.Generator | None):
    """Batches of near-equal length, so padding (and therefore cost) stays low.

    Median SMILES is 38 chars and the longest is 306; padding everything to the
    global max would triple the compute. Shuffling first, bucketing inside pools,
    then shuffling the batch order keeps the stochasticity of plain shuffling.
    """
    n = len(lens)
    idx = np.arange(n) if rng is None else rng.permutation(n)
    pool = max(bs, bs * pool_batches)
    batches = []
    for s in range(0, n, pool):
        chunk = idx[s:s + pool]
        chunk = chunk[np.argsort(lens[chunk], kind="stable")]
        for b in range(0, len(chunk), bs):
            batches.append(chunk[b:b + bs])
    if rng is not None:
        batches = [batches[i] for i in rng.permutation(len(batches))]
    return batches


def _to_batch(X, lens, idx, device):
    Lb = int(lens[idx].max())
    xb = torch.from_numpy(X[idx][:, :Lb]).to(device)
    valid = torch.from_numpy(
        np.arange(Lb)[None, :] < lens[idx][:, None]).to(device)
    return xb, valid


# --------------------------------------------------------------------------- #
# per-polymer target matrix                                                     #
# --------------------------------------------------------------------------- #

def _polymer_table(df: pd.DataFrame, with_targets: bool = True):
    """Long format -> (unique canon list, row->polymer index, Y (P,7), M (P,7)).

    A polymer contributes up to 6 of the 7 properties; the multi-task model wants
    one row per polymer with a mask over the observed heads. test_df has no
    `target` column, so pass with_targets=False for it.
    """
    canon = df["canon"].astype(str).values
    uniq, inv = np.unique(canon, return_inverse=True)
    P = len(uniq)
    if not with_targets:
        z = np.zeros((P, len(TARGETS)), dtype=np.float32)
        return uniq, inv, z, z.astype(bool)
    Y = np.zeros((P, len(TARGETS)), dtype=np.float32)
    C = np.zeros((P, len(TARGETS)), dtype=np.float32)
    tcol = {t: i for i, t in enumerate(TARGETS)}
    ti = df["target_type"].map(tcol).values
    tgt = df["target"].astype(float).values
    for p, t, v in zip(inv, ti, tgt):
        Y[p, t] += v
        C[p, t] += 1.0
    M = C > 0
    Y[M] /= C[M]                      # average if a (polymer, target) repeats
    return uniq, inv, Y, M


# --------------------------------------------------------------------------- #
# training                                                                      #
# --------------------------------------------------------------------------- #

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 31 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _lr_at(step: int, total: int, warm: int, lr: float, final_frac: float) -> float:
    if warm > 0 and step < warm:
        return lr * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    prog = min(max(prog, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * prog))
    return lr * (final_frac + (1.0 - final_frac) * cos)


def _fit_fold(Xtr, ltr, Ytr, Mtr, hp, seed, device):
    """Train one multi-task CNN. Returns (model, mu, sd) for de-standardising."""
    _seed_everything(seed)
    mu = np.zeros(len(TARGETS), dtype=np.float32)
    sd = np.ones(len(TARGETS), dtype=np.float32)
    for t in range(len(TARGETS)):
        v = Ytr[Mtr[:, t], t]
        if v.size:
            mu[t] = v.mean()
            s = v.std()
            sd[t] = s if s > 1e-8 else 1.0
    Yz = (Ytr - mu) / sd

    model = SmilesCNN(hp["vocab_size"], len(TARGETS), hp).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"],
                            weight_decay=hp["weight_decay"])
    rng = np.random.default_rng(seed)

    Yt = torch.from_numpy(Yz).to(device)
    Mt = torch.from_numpy(Mtr.astype(np.float32)).to(device)

    model.train()
    for ep in range(hp["epochs"]):
        for idx in _length_bucketed_batches(ltr, hp["batch_size"],
                                            hp["pool_batches"], rng):
            lr = _lr_at(ep, hp["epochs"], hp["warmup_epochs"],
                        hp["lr"], hp["lr_final_frac"])
            for g in opt.param_groups:
                g["lr"] = lr
            xb, valid = _to_batch(Xtr, ltr, idx, device)
            ib = torch.from_numpy(idx).to(device)
            yb, mb = Yt[ib], Mt[ib]
            pred = model(xb, valid)
            # equal weight per target, matching the unweighted-mean-R2 metric
            se = ((pred - yb) ** 2) * mb
            cnt = mb.sum(0)
            per_t = se.sum(0) / cnt.clamp(min=1.0)
            present = cnt > 0
            loss = per_t[present].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            opt.step()
            with torch.no_grad():                  # keep PAD pinned at zero
                model.emb.weight[PAD].zero_()
    return model, mu, sd


@torch.no_grad()
def _predict_all(model, X, lens, hp, device, mu, sd) -> np.ndarray:
    """-> (n, 7) in ORIGINAL units, one row per molecule."""
    model.eval()
    out = np.zeros((len(lens), len(TARGETS)), dtype=np.float64)
    for idx in _length_bucketed_batches(lens, 4 * hp["batch_size"],
                                        hp["pool_batches"], None):
        xb, valid = _to_batch(X, lens, idx, device)
        out[idx] = model(xb, valid).double().cpu().numpy()
    return out * sd.astype(np.float64) + mu.astype(np.float64)


# --------------------------------------------------------------------------- #
# the interface the driver calls                                                #
# --------------------------------------------------------------------------- #

def oof_and_test(train_df, test_df, fold_id, seeds, device=None, hp=None,
                 verbose=False):
    """Out-of-fold and test predictions from the character-level SMILES CNN.

    train_df : long format, needs columns target_type, target, canon
    test_df  : needs columns target_type, canon
    fold_id  : int array (len(train_df),), fold per train row, grouped by polymer
    seeds    : list[int]; a model is trained per (seed, fold) and predictions are
               averaged over seeds
    returns  : (oof, test_pred) float64 1-D of len(train_df) / len(test_df)

    Reads only df['canon'] -- there is no feature matrix.
    """
    cfg = dict(HP)
    if hp:
        cfg.update(hp)
    if device is None:
        # MPS is deliberately excluded: unreliable for training on this box.
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    fold_id = np.asarray(fold_id).astype(int)
    if len(fold_id) != len(train_df):
        raise ValueError("fold_id must have one entry per train row")
    seeds = list(seeds) if seeds is not None else [42]
    if not seeds:
        raise ValueError("seeds must be a non-empty list")

    uniq_tr, row2p_tr, Y, M = _polymer_table(train_df)
    uniq_te, row2p_te, _, _ = _polymer_table(test_df, with_targets=False)

    vocab = build_vocab(uniq_tr)                    # training data only
    cfg["vocab_size"] = len(vocab) + 2
    Xtr_all, ltr_all = encode(uniq_tr, vocab, cfg["max_len"])
    Xte, lte = encode(uniq_te, vocab, cfg["max_len"])

    # A polymer is out-of-fold for fold f if ANY of its rows is in fold f. Then
    # the training pool for f is every polymer with no row in f, which is
    # leak-free even if fold_id were not perfectly grouped.
    folds = np.unique(fold_id)
    tcol = {t: i for i, t in enumerate(TARGETS)}
    row_t_tr = train_df["target_type"].map(tcol).values.astype(int)
    row_t_te = test_df["target_type"].map(tcol).values.astype(int)
    y_true = train_df["target"].astype(float).values

    oof_sum = np.zeros(len(train_df))
    oof_cnt = np.zeros(len(train_df))
    test_sum = np.zeros(len(test_df))
    test_cnt = 0.0

    for seed in seeds:
        for f in folds:
            held = np.unique(row2p_tr[fold_id == f])
            in_val = np.zeros(len(uniq_tr), dtype=bool)
            in_val[held] = True
            tr_p = np.where(~in_val)[0]
            if len(tr_p) < 20 or not M[tr_p].any():
                continue

            model, mu, sd = _fit_fold(
                Xtr_all[tr_p], ltr_all[tr_p], Y[tr_p], M[tr_p],
                cfg, seed=int(seed) * 10007 + int(f), device=device)

            # --- OOF for this fold's rows ---
            val_rows = np.where(fold_id == f)[0]
            vp = np.unique(row2p_tr[val_rows])
            pv = _predict_all(model, Xtr_all[vp], ltr_all[vp], cfg, device, mu, sd)
            pos = {p: i for i, p in enumerate(vp)}
            take = np.array([pos[p] for p in row2p_tr[val_rows]])
            vals = pv[take, row_t_tr[val_rows]]
            if cfg["clip_to_train_range"]:
                vals = _clip(vals, row_t_tr[val_rows], Y[tr_p], M[tr_p])
            oof_sum[val_rows] += vals
            oof_cnt[val_rows] += 1.0

            # --- test, averaged over every (seed, fold) model ---
            pt = _predict_all(model, Xte, lte, cfg, device, mu, sd)
            tvals = pt[row2p_te, row_t_te]
            if cfg["clip_to_train_range"]:
                tvals = _clip(tvals, row_t_te, Y[tr_p], M[tr_p])
            test_sum += tvals
            test_cnt += 1.0

            if verbose:
                m = fold_id == f
                ss = ((y_true[m] - y_true[m].mean()) ** 2).sum()
                r = 1 - ((y_true[m] - vals) ** 2).sum() / ss if ss > 0 else float("nan")
                print(f"    seed={seed} fold={f} n_val={m.sum()} pooled_r2={r:+.3f}")

            del model

    if (oof_cnt == 0).any():
        raise RuntimeError("some train rows never landed in a validation fold")
    oof = oof_sum / oof_cnt
    if test_cnt == 0:
        raise RuntimeError("no fold produced a test prediction")
    test_pred = test_sum / test_cnt

    # fill any target the folds never saw with the global train mean
    for t, i in tcol.items():
        seen = M[:, i].any()
        if not seen:
            g = float(y_true[row_t_tr == i].mean()) if (row_t_tr == i).any() else 0.0
            oof[row_t_tr == i] = g
            test_pred[row_t_te == i] = g

    oof = np.asarray(oof, dtype=np.float64)
    test_pred = np.asarray(test_pred, dtype=np.float64)
    if not np.isfinite(oof).all():
        raise AssertionError("cnn produced non-finite OOF predictions")
    if not np.isfinite(test_pred).all():
        raise AssertionError("cnn produced non-finite test predictions")
    return oof, test_pred


def _clip(vals, tidx, Ytr, Mtr):
    """Clamp to the fold's observed range widened by 5%. Guards against a head
    extrapolating wildly on an unseen motif -- one bad row costs real R2 when a
    target only has ~220 of them."""
    out = np.asarray(vals, dtype=np.float64).copy()
    for t in np.unique(tidx):
        v = Ytr[Mtr[:, t], t]
        if v.size < 2:
            continue
        lo, hi = float(v.min()), float(v.max())
        pad = 0.05 * (hi - lo)
        m = tidx == t
        out[m] = np.clip(out[m], lo - pad, hi + pad)
    return out
