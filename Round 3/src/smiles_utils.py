"""Polymer SMILES (PSMILES) canonicalisation and rewriting.

Every polymer in this competition is written as a repeat unit with EXACTLY two
`*` attachment points, e.g. `*OC(=O)c1ccc(cc1)C(=O)OCC(C(C*)CCC)CCC`.

Three ways the same polymer can be written differently:

  permutational  atom ordering inside the SMILES string          -> killed by canonicalize()
  translational  the repeat unit is cut at a different bond      -> translate() builds these
  repetition     monomer vs dimer vs trimer                      -> build_oligomer() builds these

MEASURED ON THIS DATASET (train+test, 10,605 unique raw SMILES):
  * RDKit canonicalisation collapses 10,605 raw -> 8,990 distinct polymers.
  * Grouping instead by the translation-invariant macrocycle key yields only
    7 groups that contain more than one distinct canonical SMILES, and NONE of
    those 7 differ in heavy-atom count -- i.e. the dataset contains no genuine
    oligomer (monomer/dimer/trimer) duplicates at all.

  => canonicalize() is the correct dedup/group key. macrocycle_key() must NOT be
     used to merge training rows: on the 7 borderline groups it merges polymers
     that are plausibly distinct, and it buys nothing. It exists for the
     invariance AUDIT only.
"""
from __future__ import annotations

from functools import lru_cache

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

STAR = "*"


# --------------------------------------------------------------------------- #
# canonical identity                                                          #
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=300_000)
def canonicalize(smi: str) -> str:
    """RDKit canonical SMILES. The group key for CV and the dedup key.

    Falls back to the raw string when RDKit cannot parse (never happens on this
    dataset -- all 10,605 unique SMILES parse -- but the export must not crash
    on a malformed test row).
    """
    if not isinstance(smi, str) or not smi:
        return smi
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    try:
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return smi


canonical_key = canonicalize  # alias: same thing, clearer at call sites in cv.py


def attachment_points(mol) -> tuple[list[int], list[int]] | None:
    """(star_indices, neighbour_indices) for a 2-star PSMILES, else None."""
    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2:
        return None
    nbrs = []
    for s in stars:
        n = [x.GetIdx() for x in mol.GetAtomWithIdx(s).GetNeighbors()]
        if len(n) != 1:
            return None
        nbrs.append(n[0])
    return stars, nbrs


@lru_cache(maxsize=300_000)
def macrocycle_key(smi: str) -> str | None:
    """Translation-invariant key: bond the two attachment points into a ring.

    AUDIT ONLY -- see the module docstring. Returns None when the construction
    is degenerate: the two neighbours are already bonded (a 2-atom backbone such
    as `*CC*`, 1,527 of 10,605 here) or both stars hang off the same atom
    (`*C(*)R`, 13 here).
    """
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    ap = attachment_points(m)
    if ap is None:
        return None
    stars, (a, b) = ap
    if a == b or m.GetBondBetweenAtoms(a, b) is not None:
        return None
    em = Chem.RWMol(m)
    em.AddBond(a, b, Chem.BondType.SINGLE)
    for s in sorted(stars, reverse=True):
        em.RemoveAtom(s)
    try:
        mm = em.GetMol()
        Chem.SanitizeMol(mm)
        return Chem.MolToSmiles(mm)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# rewritings -- used to build the invariance certificate                      #
# --------------------------------------------------------------------------- #

def randomize(smi: str, seed: int = 0) -> str:
    """Permutational rewriting: identical graph, different atom order."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    n = m.GetNumAtoms()
    try:
        order = list(range(n))
        rng = _rng(seed)
        rng.shuffle(order)
        m2 = Chem.RenumberAtoms(m, order)
        return Chem.MolToSmiles(m2, canonical=False)
    except Exception:
        return smi


def _rng(seed: int):
    import random
    return random.Random(seed)


def build_oligomer(smi: str, n: int = 2) -> str:
    """Repetition rewriting: join n copies head-to-tail into an n-mer.

    `*A*` -> `*AA*` for n=2. Returns the input unchanged if the join is not
    possible. The result is a real RDKit-constructed molecule, not string
    concatenation -- string concatenation produces invalid SMILES for anything
    with ring-closure digits or branches.
    """
    if n < 2:
        return smi
    base = Chem.MolFromSmiles(smi)
    if base is None or attachment_points(base) is None:
        return smi
    out = base
    for _ in range(n - 1):
        out = _join(out, base)
        if out is None:
            return smi
    try:
        return Chem.MolToSmiles(out)
    except Exception:
        return smi


def _join(left, right):
    """Bond left's second attachment point to right's first, dropping both stars."""
    combo = Chem.RWMol(Chem.CombineMols(left, right))
    nL = left.GetNumAtoms()
    apL, apR = attachment_points(left), attachment_points(right)
    if apL is None or apR is None:
        return None
    (lstars, lnbrs), (rstars, rnbrs) = apL, apR
    # left keeps star[0] as the new head; right keeps star[1] as the new tail.
    drop = [lstars[1], rstars[0] + nL]
    a, b = lnbrs[1], rnbrs[0] + nL
    if a == b or combo.GetBondBetweenAtoms(a, b) is not None:
        return None
    combo.AddBond(a, b, Chem.BondType.SINGLE)
    for idx in sorted(drop, reverse=True):
        combo.RemoveAtom(idx)
    try:
        m = combo.GetMol()
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def translate(smi: str, k: int = 1) -> str:
    """Translational rewriting: cut the repeat unit at a different backbone bond.

    The repeat unit `*A-B-C*` and `*B-C-A*` describe the same infinite polymer;
    only the choice of cut point differs. This builds an alternative cut:
    it walks the backbone path between the two attachment points in the ORIGINAL
    molecule, closes the polymer into a macrocycle, then re-opens it at the k-th
    admissible backbone bond.

    A bond is admissible if it is single and not part of a ring in the original
    molecule -- cutting a ring bond would change the chemistry, not the cut point.
    Returns the input unchanged when no alternative cut exists (~13% of this
    dataset: two-atom backbones such as `*C(C)=C(*)CCC` where the attachment
    neighbours are already bonded, and fully-aromatic backbones where every
    backbone bond is in a ring).
    """
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    ap = attachment_points(m)
    if ap is None:
        return smi
    stars, (a, b) = ap
    if a == b:
        return smi

    # Backbone path in the ORIGINAL molecule. In the macrocycle this path is the
    # long way round; the shortest path there is the join bond we are about to add.
    try:
        path = Chem.GetShortestPath(m, a, b)
    except Exception:
        return smi
    if not path:
        return smi

    bonds = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    admissible = [
        (x, y) for x, y in bonds
        if (bd := m.GetBondBetweenAtoms(x, y)) is not None
        and bd.GetBondType() == Chem.BondType.SINGLE
        and not bd.IsInRing()
    ]
    if not admissible:
        return smi

    em = Chem.RWMol(m)
    if m.GetBondBetweenAtoms(a, b) is None:
        em.AddBond(a, b, Chem.BondType.SINGLE)
    for s in sorted(stars, reverse=True):
        em.RemoveAtom(s)
    try:
        ring = em.GetMol()
        Chem.SanitizeMol(ring)
    except Exception:
        return smi

    def remap(i):  # indices shift down by the number of removed stars below i
        return i - sum(1 for s in stars if s < i)

    x, y = admissible[k % len(admissible)]
    x, y = remap(x), remap(y)
    em2 = Chem.RWMol(ring)
    if em2.GetBondBetweenAtoms(x, y) is None:
        return smi
    em2.RemoveBond(x, y)
    for at in (x, y):
        s = em2.AddAtom(Chem.Atom(0))
        em2.AddBond(at, s, Chem.BondType.SINGLE)
    try:
        mm = em2.GetMol()
        Chem.SanitizeMol(mm)
        return Chem.MolToSmiles(mm)
    except Exception:
        return smi


def rewritings(smi: str, n_random: int = 3) -> dict[str, str]:
    """All rewritings of one polymer, labelled by the invariance they probe."""
    out = {"canonical": canonicalize(smi)}
    for i in range(n_random):
        out[f"permutational_{i}"] = randomize(smi, seed=i)
    out["translational"] = translate(smi, k=1)
    out["dimer"] = build_oligomer(smi, 2)
    out["trimer"] = build_oligomer(smi, 3)
    return out
