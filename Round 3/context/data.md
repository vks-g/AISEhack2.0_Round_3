# Dataset Description

**Files:** 6 files
**Size:** 378.15 MB
**Type:** csv, ipynb
**License:** Subject to Competition Rules

The training dataset contains **7,409 polymer property measurements** spanning **seven distinct polymer properties**. Each sample consists of a polymer represented by its **SMILES** string, the corresponding property value, and a **target_type** indicating which property is being predicted.

The seven target properties are:

1. **Chain Bandgap (Egc)**
2. **Bulk Bandgap (Egb)**
3. **Ionisation Energy (Ei)**
4. **Dielectric Constant (EPS)**
5. **Electron Affinity (Eea)**
6. **Refractive Index (Nc)**
7. **Glass Transition Temperature (Tg)**

## Leaderboard and Qualification

Competition rankings will be based on predictions made on a **hidden private test set**, which serves as the official evaluation dataset.

During the competition, participants will receive feedback through a **public leaderboard**, computed using a subset of the test data. The **final leaderboard** will be generated using the remaining hidden test samples and will determine the official rankings.

## Submission Format and Baseline

A **sample submission file** is provided to demonstrate the required submission format.

To help participants get started, a **baseline notebook** is also provided that demonstrates an end-to-end machine learning workflow. The baseline model:

- Generates molecular descriptors from polymer SMILES using the **RDKit** library.
- Performs basic feature engineering and preprocessing.
- Trains a **Ridge Regression** model.
- Generates predictions and creates a valid `submission.csv` file.

---

## Files

### `train.csv`
Training dataset containing **7,409 polymer property measurements**.

| Column | Description |
|---|---|
| `smiles` | SMILES representation of the polymer structure |
| `target` | Value of one of the seven polymer properties |
| `target_type` | Property category corresponding to the target value |

### `test.csv`
Test dataset used for prediction, containing **4,497 data points**.

| Column | Description |
|---|---|
| `id` | Unique sample identifier |
| `smiles` | SMILES representation of the polymer structure |
| `target_type` | Property to be predicted |

### `smile_r3.csv`
An additional auxiliary dataset containing approximately **5.97 million unique molecular SMILES**. This dataset does not contain polymer property targets and may optionally be used for:

- Self-supervised or unsupervised pretraining
- Molecular representation learning
- Feature enrichment
- Transfer learning
- Other auxiliary learning approaches

| Column | Description |
|---|---|
| `smiles` | Molecular SMILES string |

### `sample_submission.csv` (91 B)
Example submission file illustrating the required prediction format.

Sample data (id, target):

| id | target |
|---|---|
| 1 | 273.5 |
| 2 | 195.0 |
| 3 | 44.0 |
| 4 | 45.0 |
| 5 | 67.0 |
| 6 | 1.9942 |
| 7 | 5.9072 |
| 8 | -32.0 |
| 9 | 158.17 |
| 10 | 260.0 |

### `baseline_model.ipynb`
A baseline notebook demonstrating molecular descriptor generation using **RDKit**, feature preprocessing, Ridge Regression model training, and generation of a valid `submission.csv`.

---

## Data Explorer — Files list (6 files, 10 columns total)

- `PI1M.csv`
- `base_line_model.ipynb`
- `sample_submission.csv`
- `smile_r3.csv`
- `test.csv`
- `train.csv`

## License

Subject to Competition Rules
