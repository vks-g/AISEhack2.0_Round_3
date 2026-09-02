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
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdFingerprintGenerator

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
    "anhydride": "[CX3](=[OX1])[OX2][CX3](=[OX1])",
    "carbonate": "[OX2][CX3](=[OX1])[OX2]",
    "thioether_ar": "[c][SX2][c]", "sulfone_ar": "[c][SX4](=O)(=O)[c]",
    "biphenyl": "[c]1[c][c][c]([c][c]1)-[c]1[c][c][c][c][c]1",
    "fused_ring": "[R2]", "quaternary_C": "[CX4]([#6])([#6])([#6])[#6]",
    "ether_ar": "[c][OX2][#6]", "amide_ar": "[c][NX3][CX3]=[OX1]",
    "cf3": "[CX4](F)(F)F", "nitrile_ar": "[c][CX2]#[NX1]",
}
_GROUP_PATTERNS = [(k, Chem.MolFromSmarts(v)) for k, v in GROUP_SMARTS.items()]

POLYMER_TERMS = [
    "backbone_len", "heavy_atoms", "aromatic_atoms", "n_rings", "n_hetero",
    "aromatic_ratio", "hetero_ratio", "backbone_ratio",
    # backbone / side-chain decomposition. The backbone is the shortest path
    # between the two attachment points; everything hanging off it is side chain.
    # Backbone stiffness against side-chain bulk is the textbook driver of the
    # glass transition, and no whole-molecule descriptor expresses it.
    "sidechain_atoms", "sidechain_ratio", "max_sidechain_len", "n_branches",
    "backbone_rings", "backbone_rot", "backbone_arom", "backbone_hetero",
]

# Elements worth counting separately in this chemistry.
ELEMENTS = [6, 7, 8, 9, 14, 15, 16, 17, 35, 53]

# Extensive RDKit descriptors get an INTENSIVE twin (value per heavy atom).
# This is a repetition-invariance choice as much as a chemical one: an extensive
# quantity DOUBLES when a repeat unit is written as its dimer, an intensive one
# does not. Adding the intensive twin gives the fingerprint models a view of the
# molecule that does not move under the rewriting the host asked about.
INTENSIVE_OF = [
    "MolWt", "HeavyAtomCount", "NumRotatableBonds", "NumHAcceptors",
    "NumHDonors", "RingCount", "NumAromaticRings", "TPSA", "LabuteASA",
    "MolMR", "NumValenceElectrons", "NHOHCount", "NOCount", "MolLogP",
]

CHARGE_TERMS = ["q_min", "q_max", "q_mean", "q_absmean", "q_std", "q_range"]


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
        + [f"el_n{z}" for z in ELEMENTS] + [f"el_f{z}" for z in ELEMENTS]
        + [f"iv_{n}" for n in INTENSIVE_OF]
        + [f"chg_{n}" for n in CHARGE_TERMS]
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
    ha = m.GetNumHeavyAtoms()
    arom = sum(1 for a in m.GetAtoms() if a.GetIsAromatic())
    rings = m.GetRingInfo().NumRings()
    het = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() not in (1, 6, 0))

    backbone = -1.0
    bb_path: tuple = ()
    if len(stars) == 2:
        try:
            bb_path = Chem.GetShortestPath(m, stars[0], stars[1])
            backbone = float(len(bb_path) - 2)
        except Exception:
            bb_path, backbone = (), -1.0

    bb = set(bb_path) - set(stars)
    side = [a.GetIdx() for a in m.GetAtoms()
            if a.GetAtomicNum() != 0 and a.GetIdx() not in bb]
    n_side = float(len(side))
    # longest side chain = deepest excursion off the backbone
    max_side = 0.0
    n_branch = 0.0
    if bb:
        seen = set(bb)
        for b_ in bb:
            for nb_ in m.GetAtomWithIdx(b_).GetNeighbors():
                if nb_.GetIdx() in bb or nb_.GetAtomicNum() == 0:
                    continue
                n_branch += 1.0
                depth, frontier = 0, [nb_.GetIdx()]
                local = set(seen)
                while frontier and depth < 30:
                    depth += 1
                    nxt = []
                    for u in frontier:
                        local.add(u)
                        for w in m.GetAtomWithIdx(u).GetNeighbors():
                            if w.GetIdx() not in local and w.GetAtomicNum() != 0:
                                nxt.append(w.GetIdx())
                    frontier = nxt
                max_side = max(max_side, float(depth))
    bb_rings = float(sum(1 for i in bb if m.GetAtomWithIdx(i).IsInRing()))
    bb_arom = float(sum(1 for i in bb if m.GetAtomWithIdx(i).GetIsAromatic()))
    bb_het = float(sum(1 for i in bb
                       if m.GetAtomWithIdx(i).GetAtomicNum() not in (1, 6, 0)))
    bb_rot = 0.0
    for b_ in m.GetBonds():
        i, j = b_.GetBeginAtomIdx(), b_.GetEndAtomIdx()
        if i in bb and j in bb and b_.GetBondType() == Chem.BondType.SINGLE \
                and not b_.IsInRing():
            bb_rot += 1.0

    parts.append(np.array([
        backbone, ha, arom, rings, het,
        arom / max(ha, 1), het / max(ha, 1), backbone / max(ha, 1),
        n_side, n_side / max(ha, 1), max_side, n_branch,
        bb_rings, bb_rot, bb_arom, bb_het,
    ], dtype=np.float32))

    parts.append(np.array(
        [len(m.GetSubstructMatches(p)) if p is not None else 0 for _, p in _GROUP_PATTERNS],
        dtype=np.float32))

    counts = {z: 0 for z in ELEMENTS}
    for a in m.GetAtoms():
        if a.GetAtomicNum() in counts:
            counts[a.GetAtomicNum()] += 1
    parts.append(np.array([counts[z] for z in ELEMENTS], dtype=np.float32))
    parts.append(np.array([counts[z] / max(ha, 1) for z in ELEMENTS], dtype=np.float32))

    # intensive twins -- unchanged when the repeat unit is written as its dimer
    parts.append(np.array([float(d.get(n, 0.0)) / max(ha, 1) for n in INTENSIVE_OF],
                          dtype=np.float32))

    try:
        mc = Chem.Mol(m)
        AllChem.ComputeGasteigerCharges(mc)
        q = np.array([float(a.GetDoubleProp("_GasteigerCharge"))
                      for a in mc.GetAtoms()], dtype=np.float64)
        q = q[np.isfinite(q)]
        chg = ([q.min(), q.max(), q.mean(), np.abs(q).mean(), q.std(),
                q.max() - q.min()] if q.size else [0.0] * 6)
    except Exception:
        chg = [0.0] * 6
    parts.append(np.array(chg, dtype=np.float32))

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
    """Featurise a list of SMILES, in parallel when that is actually available.

    The parallel path is a convenience, not a requirement. Inside a notebook
    every function lives in `__main__`, and a spawn-based ProcessPoolExecutor
    cannot pickle it -- so this raises `PicklingError` in Jupyter on macOS and
    Windows, and only survives on Linux because fork copies the address space.
    Relying on that would make the submission notebook platform-dependent, so any
    failure falls back to a serial loop: ~200 s for all 12,345 molecules, which
    is a rounding error against the model training that follows.
    """
    n_jobs = n_jobs or max(1, (os.cpu_count() or 4) - 2)
    if n_jobs > 1 and len(uniq) > 200 and _parallel_safe():
        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                return list(ex.map(featurize_one, uniq, chunksize=32))
        except Exception as exc:      # PicklingError, BrokenProcessPool, OSError
            print(f"featurize: parallel path failed ({type(exc).__name__}); "
                  f"falling back to serial")
    return [featurize_one(s) for s in uniq]


def _parallel_safe() -> bool:
    """Whether a process pool can actually run `featurize_one`.

    With the `fork` start method (Linux, so Kaggle) the child inherits the
    address space and any function works. With `spawn` (macOS, Windows) the
    child re-imports the function by qualified name, which is impossible when it
    was defined in a notebook cell -- everything there lives in `__main__`.
    Probing that up front avoids a BrokenProcessPool and the wall of child
    tracebacks it prints before the fallback catches it.
    """
    import multiprocessing as mp
    try:
        if mp.get_start_method(allow_none=False) == "fork":
            return True
    except Exception:
        return False
    return getattr(featurize_one, "__module__", "__main__") != "__main__"


def warm_cache(*frames) -> None:
    """Featurise every molecule in the given frames up front, once."""
    allsmi = []
    for f in frames:
        allsmi.extend(f["canon"].tolist() if hasattr(f, "columns") else list(f))
    featurize(allsmi)


def family_of(col: str) -> str:
    return col.split("_", 1)[0]
