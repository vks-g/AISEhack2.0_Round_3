---
name: explainability
description: Produce the per-target feature-attribution report that is half the Round 3 judged rubric. Use when writing the explainability section of the notebook or asked which features drive a property.
---

# Explainability

The other judged Round 3 theme, and the cheaper of the two to do well.

```bash
cd "Round 3" && ./.venv/bin/python -m src.explain --config lgbm
```

## Use LightGBM's own TreeSHAP, not the `shap` package

`booster.predict(X, pred_contrib=True)` returns exact SHAP values — the same
algorithm `shap` implements, computed inside LightGBM. That means **no extra
dependency that could fail to install inside the Kaggle notebook**, which matters
because a failed import at cell 30 loses the whole run. `src/explain.py` already
does this.

## Three views, because 2,978 features is the wrong granularity

Attributing to individual Morgan bits reads as noise to a materials scientist.
The report gives:

1. **Per-target attribution by feature family** — RDKit descriptors vs Morgan r=2
   vs r=3 vs atom-pair vs torsion vs MACCS vs polymer-specific vs functional
   groups, as a percentage of total |SHAP|. This is the view that reads as
   chemistry.
2. **Top interpretable features per target** — restricted to named RDKit
   descriptors (`rd_`), polymer terms (`po_`), SMARTS functional groups (`grp_`)
   and partner properties (`true_`). These have meanings you can write a sentence
   about.
3. **Top individual features**, for completeness.

Feature-family prefixes are defined in `src/features.py`: `rd_ mfp2_ mfp3_ ap_
tt_ mac_ po_ grp_`, plus `true_`/`has_`/`n_partners` when a config adds partner
features.

## What makes the section persuasive

- Tie attributions to known polymer physics. Aromatic ratio and conjugation
  driving the bandgaps (`egc`, `egb`); backbone flexibility, rotatable bonds and
  hydrogen bonding driving `tg`; polarisability and molar refraction driving `nc`
  and `eps`. If the model's top features contradict that, say so — an honest
  surprise is better material than a tidy story.
- Show the **partner-property attribution** for the DFT block. When `true_egc`
  dominates the `ei` model, that is the physics relation `ei = egc + eea`
  appearing directly in the attribution, and it is worth pointing out explicitly.
- Report per target, not globally. The seven properties have genuinely different
  drivers and a single global chart hides that.

## Cheap additions if time allows

- Permutation importance on a held-out fold as a second, model-agnostic view.
- A partial-dependence curve for the two or three top descriptors per target.
- One worked single-molecule explanation: pick a test polymer, show its SHAP
  waterfall, and write two sentences on why the model predicted what it did.
