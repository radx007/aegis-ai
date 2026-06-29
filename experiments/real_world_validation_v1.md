# Experiment — Real World Validation V1

Date:
2026-06-29

Status:
Completed

---

## Objective

Validate whether the baseline model generalizes to recordings outside ESC50.

---

## Setup

Model:
YAMNet + Logistic Regression

Training Dataset:
ESC50

Validation Dataset:
Custom recordings

Classes:
- clock_alarm
- crying_baby
- door_wood_knock
- siren

---

## Results

Real World Accuracy:

94.87%

Cross Validation:

99.38%

Difference:

-4.5%

---

## Observations

Model remained stable.

Generalization appears good.

Performance dropped slightly outside training data.

Expected behavior.

---

## Risks

Small validation dataset.

Environmental diversity limited.

---

## Decision

Baseline accepted.

Proceed to packaging and deployment preparation.