# Experiment — Evaluation V1

Date:
2026-06-27

Status:
Completed

---

## Objective

Validate whether the baseline classifier generalizes beyond a single train/test split.

---

## Setup

Dataset:
ESC50

Selected Classes:
- clock_alarm
- crying_baby
- door_wood_knock
- siren

Feature Extractor:
YAMNet

Classifier:
Logistic Regression

Validation:
5-Fold Cross Validation

---

## Results

Cross Validation Scores:

- Fold 1 → 1.0000
- Fold 2 → 1.0000
- Fold 3 → 0.9688
- Fold 4 → 1.0000
- Fold 5 → 1.0000

Mean:

0.9938

Classification Report:

Precision:
1.00

Recall:
1.00

F1:
1.00

---

## Observations

- Performance remained consistent across folds.
- One fold showed a minor decrease.
- No class confusion observed.
- Selected classes appear easily separable.

---

## Risks

- Dataset is limited in size.
- Evaluation performed only on ESC50.
- Real-world performance is unknown.

---

## Decision

Accept baseline temporarily.

Proceed to real-world validation.

Model is not considered production-ready yet.