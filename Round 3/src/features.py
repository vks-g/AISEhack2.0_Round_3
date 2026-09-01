"""Molecular featurisation, keyed by CANONICAL SMILES.

Feature families and their column prefixes (the prefix is what the
explainability report groups by):

    rd_    RDKit physicochemical descriptors      217
    mfp2_  Morgan fingerprint, radius 2          1024
    mfp3_  Morgan fingerprint, radius 3           512
    ap_    atom-pair fingerprint                  512
    tt_    topological torsion fingerprint        512
    mac_   MACCS keys                             167
    po_    polymer-specific terms                   8
    grp_   SMARTS functional-group counts          26
                                                 ----
                                                 2978

Everything is computed from the canonical SMILES, which is what makes the
pipeline invariant to how a polymer was written (see src/smiles_utils).

CACHING: `featurize()` memoises to `.cache/` so local CV iterates in seconds.
That cache is LOCAL ONLY. Competition rule 6.2.4 forbids shipping cached
feature files, so the exported notebook always recomputes from scratch
(~19 s wall for all 10,605 molecules on 9 processes -- it is not a bottleneck).
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

from src.paths import cache_dir

MFP2_BITS, MFP3_BITS, AP_BITS, TT_BITS = 1024, 512, 512, 512

_DESC_NAMES = [n for n, _ in Descriptors._descList]
_F32MAX = float(np.finfo(np.float32).max)

GROUP_SMARTS = {
    "aromatic_6": "[a]1[a][a][a][a][a]1", "aromatic_5": "[a]1[a][a][a][a]1",
    "amide": "[NX3][CX3](=[OX1])", "ester": "[CX3](=[OX1])[OX2]",
    "ether": "[OD2]([#6])[#6]", "hydroxyl": "[OX2H]", "carbonyl": "[CX3]=[OX1]",
    "carboxyl": "[CX3](=[OX1])[OX2H1]", "sulfonyl": "[#16X4](=[OX1])(=[OX1])",
    "sulfide": "[#16X2H0]", "nitrile": "[NX1]#[CX2]", "nitro": "[$([NX3](=O)=O)]",
    "amine_1": "[NX3;H2][#6]", "amine_2": "[NX3;H1]([#6])[#6]",
    "amine_3": "[NX3]([#6])([#6])[#6]", "imide": "[NX3](C=O)C=O",
    "urethane": "[NX3][CX3](=[OX1])[OX2]", "urea": "[NX3][CX3](=[OX1])[NX3]",
    "halide_F": "[F]", "halide_Cl": "[Cl]", "halide_Br": "[Br]", "halide_I": "[I]",
    "siloxane": "[Si][OX2][Si]", "phosphate": "[PX4](=[OX1])",
    "alkene": "[CX3]=[CX3]", "alkyne": "[CX2]#[CX2]",
}
_GROUP_PATTERNS = [(k, Chem.MolFromSmarts(v)) for k, v in GROUP_SMARTS.items()]

POLYMER_TERMS = [
    "backbone_len", "heavy_atoms", "aromatic_atoms", "n_rings", "n_hetero",
    "aromatic_ratio", "hetero_ratio", "backbone_ratio",
]


def feature_names() -> list[str]:
    return (
        [f"rd_{n}" for n in _DESC_NAMES]
        + [f"mfp2_{i}" for i in range(MFP2_BITS)]
        + [f"mfp3_{i}" for i in range(MFP3_BITS)]
        + [f"ap_{i}" for i in range(AP_BITS)]
        + [f"tt_{i}" for i in range(TT_BITS)]
        + [f"mac_{i}" for i in range(167)]
        + [f"po_{n}" for n in POLYMER_TERMS]
        + [f"grp_{k}" for k in GROUP_SMARTS]
    )


N_FEATURES = len(feature_names())

_GEN = {}


def _gens():
    if not _GEN:
        _GEN["mfp2"] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=MFP2_BITS)
        _GEN["mfp3"] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=MFP3_BITS)
        _GEN["ap"] = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=AP_BITS)
        _GEN["tt"] = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=TT_BITS)
    return _GEN


def featurize_one(smi: str) -> np.ndarray:
    """Feature vector for one SMILES. Returns zeros if RDKit cannot parse it."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return np.zeros(N_FEATURES, dtype=np.float32)
    g = _gens()
    parts = []

    d = Descriptors.CalcMolDescriptors(m)
    # Ipc and a few graph descriptors can exceed the float32 range; clip before the
    # cast so the value saturates instead of silently becoming inf.
    desc = np.array([d.get(k, 0.0) for k in _DESC_NAMES], dtype=np.float64)
    desc = np.nan_to_num(desc, nan=0.0, posinf=_F32MAX, neginf=-_F32MAX)
    parts.append(np.clip(desc, -_F32MAX, _F32MAX).astype(np.float32))
    for key in ("mfp2", "mfp3", "ap", "tt"):
        parts.append(g[key].GetFingerprintAsNumPy(m).astype(np.float32))
    mac = np.zeros(167, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m), mac)
    parts.append(mac)

    stars = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    backbone = -1.0
    if len(stars) == 2:
        try:
            backbone = float(len(Chem.GetShortestPath(m, stars[0], stars[1])) - 2)
        except Exception:
            backbone = -1.0
    ha = m.GetNumHeavyAtoms()
    arom = sum(1 for a in m.GetAtoms() if a.GetIsAromatic())
    rings = m.GetRingInfo().NumRings()
    het = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() not in (1, 6, 0))
    parts.append(np.array([
        backbone, ha, arom, rings, het,
        arom / max(ha, 1), het / max(ha, 1), backbone / max(ha, 1),
    ], dtype=np.float32))

    parts.append(np.array(
        [len(m.GetSubstructMatches(p)) if p is not None else 0 for _, p in _GROUP_PATTERNS],
        dtype=np.float32))

    v = np.concatenate(parts)
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


_MEMO: dict[str, np.ndarray] = {}
_DISK = None


def _disk_paths():
    d = cache_dir()
    return d / f"features_v{N_FEATURES}.npy", d / f"features_v{N_FEATURES}.keys.json"


def _load_disk() -> None:
    """Warm the in-process memo from disk once per interpreter.

    Stored as a plain .npy plus a key list: uncompressed so the load is a memory
    map rather than a decompress, which keeps repeated CV runs near-instant.
    """
    global _DISK
    if _DISK is not None:
        return
    _DISK = True
    mat_p, key_p = _disk_paths()
    if not (mat_p.exists() and key_p.exists()):
        return
    try:
        keys = json.loads(key_p.read_text())
        mat = np.load(mat_p, mmap_mode="r")
        if len(keys) != mat.shape[0] or mat.shape[1] != N_FEATURES:
            return
        for i, k in enumerate(keys):
            _MEMO.setdefault(k, np.asarray(mat[i]))
    except Exception:
        pass


def _save_disk() -> None:
    if not _MEMO:
        return
    mat_p, key_p = _disk_paths()
    keys = list(_MEMO.keys())
    mat = np.vstack([_MEMO[k] for k in keys])
    tmp = mat_p.with_name(mat_p.name + ".tmp.npy")
    np.save(tmp, mat)
    tmp.replace(mat_p)
    key_p.write_text(json.dumps(keys))


def featurize(smiles, n_jobs: int | None = None, use_cache: bool = True) -> np.ndarray:
    """(len(smiles), N_FEATURES) float32. `smiles` must already be canonical.

    Memoised PER MOLECULE, not per call, so cross-validation folds -- which each
    pass a different subset -- all hit the cache after the first pass.
    Set use_cache=False to force recomputation (what the exported notebook does).
    """
    smiles = [str(s) for s in smiles]
    if not use_cache:
        uniq = list(dict.fromkeys(smiles))
        rows = _compute(uniq, n_jobs)
        idx = {s: i for i, s in enumerate(uniq)}
        return np.vstack(rows)[[idx[s] for s in smiles]]

    _load_disk()
    uniq = list(dict.fromkeys(smiles))
    missing = [s for s in uniq if s not in _MEMO]
    if missing:
        for s, v in zip(missing, _compute(missing, n_jobs)):
            _MEMO[s] = v
        try:
            _save_disk()
        except Exception:
            pass
    return np.vstack([_MEMO[s] for s in smiles])


def _compute(uniq: list[str], n_jobs: int | None):
    n_jobs = n_jobs or max(1, (os.cpu_count() or 4) - 2)
    if n_jobs > 1 and len(uniq) > 200:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            return list(ex.map(featurize_one, uniq, chunksize=32))
    return [featurize_one(s) for s in uniq]


def warm_cache(*frames) -> None:
    """Featurise every molecule in the given frames up front, once."""
    allsmi = []
    for f in frames:
        allsmi.extend(f["canon"].tolist() if hasattr(f, "columns") else list(f))
    featurize(allsmi)


def family_of(col: str) -> str:
    return col.split("_", 1)[0]
