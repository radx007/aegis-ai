# Experiment — Baseline Classifier V1

Date:
2026-06-27

Objective:
Train first classifier using YAMNet embeddings.

Dataset:
ESC50

Selected Classes:
- alarm
- baby
- siren
- knock

Features:
YAMNet embeddings

Classifier:
Logistic Regression

Train/Test:
80/20

Results:

Accuracy:
1.00

Observations:
- No visible prediction errors
- Result appears unusually high
- Possible leakage or easy separation

Decision:
Do NOT trust result yet.

Next Steps:
- Cross validation
- Confusion matrix
- Real-world validation