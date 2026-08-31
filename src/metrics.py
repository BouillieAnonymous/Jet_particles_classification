"""Evaluation helpers for binary jet classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if labels.shape != probabilities.shape:
        raise ValueError("Labels and probabilities must have the same shape")
    if np.unique(labels).size != 2:
        raise ValueError("ROC-AUC requires both classes")
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
    }
