---
name: invariance-audit
description: Measure and evidence polymer representation invariance — that the same polymer written differently gets the same prediction. Use for canonicalization, SMILES augmentation, test-time augmentation, or the Round 3 invariance deliverable.
---

# Polymer invariance

One of the two judged Round 3 themes. Run it:

```bash
cd "Round 3" && ./.venv/bin/python -m src.invariance --config <name> --n 200
```

## Read this before spending time here

**Measured on train+test (10,605 unique raw SMILES): the dataset contains no
genuine oligomer duplicates and only 7 borderline translational groups out of
~9,000 polymers.** RDKit canonicalisation alone collapses 10,605 raw strings to
8,990 distinct polymers, and grouping by a translation-invariant macrocycle key
merges almost nothing further.

So invariance is a **robustness certificate for the rubric, not leaderboard
points**. Build the evidence, report the numbers, and do not spend a submission
slot or a day of the two remaining on repeat-unit reduction. The previous version
of this file claimed canonicalising to "the shortest repeat unit" was "the
cheapest and biggest win" — that was never measured and is wrong.

## The three invariances, and what the pipeline actually guarantees

| kind | what changes | guarantee |
|---|---|---|
| **permutational** | atom ordering in the string | **EXACT.** Every feature derives from `canonicalize()`, which is idempotent, so predictions are bit-identical. Verified: 400/400 random rewritings canonicalise back to the same string. |
| **translational** | the repeat unit is cut at a different bond | Not exact. A different cut moves the two `*`, so descriptors and fingerprints shift. The audit measures how much. |
| **repetition** | monomer vs dimer vs trimer | Not exact, same reason — an n-mer has n times the atoms. |

The audit reports each separately, because reporting one number hides which
guarantee is real. `src.smiles_utils` builds all three rewritings with RDKit
graph surgery, verified valid on 400/400 sampled training polymers:

- `randomize(smi, seed)` — permutational
- `translate(smi, k)` — translational (works on 85.6% of polymers; the rest have
  a two-atom or fully-aromatic backbone with no alternative cut)
- `build_oligomer(smi, n)` — repetition, real graph joining, not string
  concatenation. String concatenation produces invalid SMILES for anything with
  ring-closure digits — the previous `repeat_unit()` helper did exactly that and
  emitted unparseable strings for most molecules.

## Levers, if the measured spread is too large

1. **Canonicalise on input.** Already done everywhere via `src/data.py`. This is
   what makes permutational invariance exact, and it is free.
2. **Train-time augmentation.** Add randomised writings as extra training rows.
   Keep augmented copies in the same CV fold as their parent or you leak.
3. **Test-time augmentation.** Predict over k rewritings and average. Costs k×
   inference; budget it against the Kaggle runtime.

Levers 2 and 3 are worth trying for the four small properties as pure
regularisation — but score them through `src.cv` like anything else.

## Reporting it

Put the per-kind table in the notebook. The claim to make is precise: exact
invariance to permutational rewriting by construction, plus a measured bound on
translational and repetition drift. Do not claim invariance you have not measured.
