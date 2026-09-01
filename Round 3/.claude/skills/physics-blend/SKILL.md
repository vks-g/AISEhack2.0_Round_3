---
name: physics-blend
description: Apply the inter-property physics relations correctly — affine-fitted, never as raw identities — and tune the blend weight without leaking. Use when working on ei, eea, egb, eps or nc.
---

# Physics relations between the DFT properties

Five relations, **measured on co-observed train pairs**. `raw` is the textbook
identity as both shipped notebooks apply it; `fitted` is the same expression
through a least-squares `a*x + b` calibrated on the training fold:

| target | expression | n | raw R² | fitted R² | a | b |
|---|---|---|---|---|---|---|
| `ei`  | `egc + eea` | 59 | 0.9629 | 0.9650 | 1.005 | 0.013 |
| `eea` | `ei - egc` | 59 | 0.9710 | 0.9727 | 1.004 | -0.054 |
| `egb` | `egc` | 175 | 0.8922 | **0.9282** | 1.159 | -1.044 |
| `eps` | `nc ** 2` | 134 | **0.3364** | **0.8553** | 1.040 | 0.615 |
| `nc`  | `sqrt(eps)` | 134 | **0.1708** | **0.8370** | 0.887 | 0.050 |

## The single most important line in this file

**Always fit. Never apply the raw identity.**

Look at `eps`. The Maxwell relation `eps = n²` holds for the *optical* dielectric
constant; the measured *static* `eps` sits well above it. The correlation is
0.925, so the shape is right — but the raw identity is badly biased and scores
0.336, while the same expression with a fitted offset scores 0.855. Same for
`nc`: 0.171 raw, 0.837 fitted. `egb` gains 0.036 from fitting too.

Both `submissions/*.ipynb` apply the raw form. The parent CLAUDE.md's old table
listed "eps ≈ nc², R² 0.843" — that number only ever held affine-fitted, and it
was being used to justify the unfitted version.

## Coverage — this only reaches rows with a measured partner

Fraction of each property's **test** rows whose partner values are present in
train.csv:

    ei 55/148 (37%)   eea 51/147 (35%)   egb 124/224 (55%)
    eps 95/153 (62%)  nc 95/153 (62%)

Rows without a partner fall through to the model prediction untouched. Two-stage
blending — true partners first, then *predicted* partners for the rest — extends
reach, but the predicted-partner population has materially worse physics R² than
the true-partner one. Fit and tune the two populations separately, or you
calibrate on a mixture and apply it to both. `submissions/round-3-aisehack.ipynb`
made exactly that mistake: one weight fitted across both populations, then raised
to 0.90 shrinkage for `ei`/`eea` so it touches 100% of those rows.

## Doing it without leaking

`src/physics.py` and `src/configs/lgbm_physics.py` implement this:

```python
rels   = physics.fit_relations(train_fold)       # calibrate on the TRAIN FOLD only
lookup = physics.wide_table(train_fold)          # partner labels from the TRAIN FOLD only
est, covered = physics.partner_estimate(df, rels, lookup)
w, _   = physics.tune_weight(y, inner_oof, est)  # weight from an INNER OOF split
pred   = physics.blend(pred, est, w)
```

Three separate leak points, all of which must use training-fold data only:
the calibration `a, b`; the partner lookup; and the blend weight. Tuning the
weight on the *outer* OOF is the classic way to get an OOF number that does not
survive contact with the leaderboard — which is a live hypothesis for why the
team's 0.903 OOF came back as 0.883 on the public LB.

## Where this is worth the most

`eps` (0.725 in the plain `lgbm` config) and `ei` (0.806) are the two weakest
targets, and physics reaches both — `eps` at 62% coverage with a fitted R² of
0.855, `ei` at 37% with 0.965. Those are the highest-value rows in the dataset.

## Staging: count the measured sources, do not just ask "all or nothing"

A binary true/predicted split throws away the partially covered rows, and they
are the largest group. For the 148 `ei` test rows: **55 have both `egc` and
`eea` measured, 69 have exactly one, and only 24 have neither.** Lumping those
69 in with the fully-predicted rows discards a real measurement and calibrates
one weight across two populations with very different reliability.

`src/oof.py apply_physics()` groups rows by the number of measured sources and
gives each group its own calibration and its own blend weight. Measured on
lgbm alone, that staging is worth **+0.063 on ei, +0.059 on eps, +0.043 on eea,
+0.031 on egb, +0.028 on nc** — mean 0.8645 → 0.8941.

The fitted weights are informative in themselves:

    ei   {2 sources: 0.81, 1: 0.75, 0: 1.00}
    eps  {1: 0.54, 0: 1.00}

A weight of 1.00 at level 0 means that for rows with no measured partner, the
relation applied to *predicted* partners beats the direct model outright. That is
not a bug: `eps` (229 rows, R² 0.78 direct) is better predicted by routing
through `nc` than by modelling it head-on.

## The trap: iterating the relations leaks, spectacularly

Because level-0 weights are so high, the obvious next step is to refine the
predicted property table by belief propagation over the relation graph before
using it. Measured naively that is worth **+0.042 mean R² (0.893 → 0.935)**.

**It is entirely fake.** `eps` is refined *from* `nc`, and `nc` is then predicted
*from* the refined `eps`, so a validation polymer's own measured label returns to
its own prediction through a two-hop loop. With the mask in place — each
(target, fold) redoes the refinement with that fold's true target values replaced
by their out-of-fold predictions — the real gain is **+0.001**.

If an idea in this family suddenly buys several points, assume a cycle in the
constraint graph before believing it. This is the same class of error that most
plausibly explains the team's 0.903 OOF → 0.883 public LB gap.

## Combining base models: measure, do not assume

With three correlated gradient-boosting models, a plain **mean beats a
cross-fitted Ridge meta-learner** (0.8961 vs 0.8950 after physics), and Ridge
over a *single* base model is worse than that model alone (0.8622 vs 0.8645).
`src/oof.py stack()` therefore defaults to the mean and only tries Ridge when
there are four or more base models, keeping it per target only where it actually
wins out-of-fold.
