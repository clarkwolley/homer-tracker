"""
Model evaluation utilities.

HR prediction is extremely imbalanced (~3% positive rate),
so accuracy is meaningless. We use AUC, precision, recall,
and Brier score to measure model quality.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)


def evaluate_model(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Evaluate a binary classification model."""
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true = np.array(y_true)

    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
        "total_samples": len(y_true),
        "actual_positives": int(y_true.sum()),
        "predicted_positives": int(y_pred.sum()),
    }


def print_evaluation(metrics: dict, model_name: str = "Model"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*50}")
    print(f"📊 {model_name} Evaluation")
    print(f"{'='*50}")
    print(f"  ROC AUC:    {metrics['roc_auc']:.3f}  (1.0=perfect, 0.5=coin flip)")
    print(f"  Brier:      {metrics['brier_score']:.3f}  (lower is better)")
    print(f"  Precision:  {metrics['precision']:.3f}  (when it says 'HR', how often correct?)")
    print(f"  Recall:     {metrics['recall']:.3f}  (of all HRs, how many caught?)")
    print(f"  F1 Score:   {metrics['f1']:.3f}")
    print(f"  Threshold:  {metrics['threshold']}")
    print(f"  Samples:    {metrics['total_samples']} total, "
          f"{metrics['actual_positives']} actual HRs, "
          f"{metrics['predicted_positives']} predicted HRs")
    print(f"{'='*50}")
