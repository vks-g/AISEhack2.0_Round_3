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
