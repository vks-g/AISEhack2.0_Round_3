# AISEHack 2.0 Polymer Property Prediction: Round 3

**Competition Host:** Aryan Mishra
**Type:** Community Prediction Competition — Private — 3 days to go (at time of capture)
**Round:** Round 3

**Prizes & Awards:** Kudos (Does not award Points or Medals)

**Participation (at time of capture):**
- 17 Entrants
- 15 Participants
- 11 Teams
- 71 Submissions

**Tags:** Custom Metric

**Competition Website:** https://www.kaggle.com/competitions/aisehack-2-0/

---

## Overview

Welcome to the Final Stage of AISEHack 2.0 — Polymer Property Prediction, Round 3!

Having built strong predictive baselines in Round 2, Round 3 raises the bar. The challenge moves beyond raw accuracy to robustness and trust — two qualities that separate a research prototype from a model that materials scientists can actually rely on.

Participants continue working with the same seven target properties spanning thermal, electronic, optical, and structural behavior, using the same core datasets from Round 2.

### Additional Auxiliary Dataset

This round introduces an additional molecular SMILES dataset containing approximately **5.97 million unique molecular structures**. The dataset contains SMILES representations of molecules and does **not** include the target polymer properties.

Participants are encouraged to explore this auxiliary dataset for approaches such as:
- Self-supervised or unsupervised pretraining of molecular representations
- SMILES-based representation learning
- Feature enrichment and molecular embedding generation
- Transfer learning to improve polymer property prediction models

The use of the auxiliary dataset is optional, and participants are free to determine how, or whether, it can improve their modelling pipeline.

### Two Central Themes for This Round

1. **Explainability of the models**
2. **Polymer invariance**

The goal is not only to build accurate models, but also models whose predictions are interpretable, robust, and invariant to different valid representations of the same polymer structure.

### Timeline (as shown on Overview page)
- **Start:** 7 days ago
- **Close:** 3 days to go

---

## Description

Congratulations on qualifying for **Round 3** of the Polymer Property Prediction Challenge!

In the previous rounds, participants developed machine learning models to predict polymer properties from polymer structures represented as **SMILES** strings. In Round 3, the challenge continues with the prediction of **seven fundamental polymer properties**, while placing greater emphasis on **robustness, explainability, and polymer invariance**.

### Task: Predict the following seven polymer properties

| Property | Description |
|---|---|
| **Chain Bandgap (Egc)** | Electronic bandgap of an isolated polymer chain. |
| **Bulk Bandgap (Egb)** | Electronic bandgap of the polymer in the bulk phase. |
| **Ionisation Energy (Ei)** | Energy required to remove an electron from the polymer. |
| **Electron Affinity (Eea)** | Energy released when the polymer accepts an electron. |
| **Dielectric Constant (EPS)** | Ability of the polymer to store electrical energy in an electric field. |
| **Refractive Index (Nc)** | Optical property describing how light propagates through the polymer. |
| **Glass Transition Temperature (Tg)** | Temperature at which the polymer transitions from a glassy to a rubbery state. |

In addition to the core datasets used in Round 2, an additional auxiliary dataset containing approximately **5.97 million unique molecular SMILES** is provided. Participants may optionally use this dataset for pretraining, representation learning, feature enrichment, or other auxiliary learning approaches.

The goal is not only to develop accurate models, but also models that are **robust, generalizable, interpretable, and invariant** to different valid representations of the same polymer structure.

We look forward to your innovative solutions and wish you the best of luck in **Round 3**!

---

## Evaluation

The evaluation metric for this competition is the **mean coefficient of determination R²** across the seven targets:

```
Score = (R²_Tg + R²_Egc + R²_Egb + R²_Ei + R²_Eea + R²_Nc + R²_EPS) / 7
```

where R² is defined as:

```
R² = 1 - [ Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)² ]
```

where:
- yᵢ = ground truth
- ŷᵢ = predictions
- ȳ = mean of ground truth values

---

## Submission File

The submission file for this competition must be a **CSV file**. For each `id` in the test set, you must predict based on the `target_type`. The file should contain a header and have the following format:

```csv
id,target
1,220
2,2.3
3,110
4,70
```

---

## Timeline

- **Start Date:** 22 August 2026
- **Final Submission Deadline:** 3 September 2026

---

## Requirements

- Submissions to this competition must be made through Notebooks. Please ensure the notebooks are shared with the competition hosts. In order for the "Submit" button to be active after a commit, the following conditions must be met:
- For this competition, publicly available external data is **not allowed**, including pre-trained models. Instead, participants can use the auxiliary data provided in the Data section.
- Submission file must be named `submission.csv`.
