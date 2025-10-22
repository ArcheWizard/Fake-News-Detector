from __future__ import annotations

import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score


def _softmax(x: np.ndarray) -> np.ndarray:
    """Apply numerically stable softmax normalization to logits.

    Converts raw logits to probability distributions using the softmax function
    with max subtraction for numerical stability to prevent overflow.

    Args:
        x: Array of shape (batch_size, num_classes) containing raw logits

    Returns:
        Array of same shape with probabilities that sum to 1.0 along axis 1.
        Each value is in range [0, 1].

    Note:
        Uses the numerically stable formulation:
        softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        This prevents overflow with large logit values.

    Examples:
        >>> logits = np.array([[2.0, 1.0], [0.5, 1.5]])
        >>> probs = _softmax(logits)
        >>> print(probs.sum(axis=1))  # Should be [1., 1.]
        [1. 1.]
    """
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute comprehensive classification metrics for binary fake news detection.

    Calculates accuracy, precision, recall, F1-score, and ROC AUC for model
    evaluation. Designed to work as a callback for HuggingFace Trainer.

    Args:
        eval_pred: EvalPrediction object or dict containing:
            - predictions: Raw logits of shape (batch_size, num_classes)
            - label_ids: Ground truth labels of shape (batch_size,)

            Accepts both object attribute format (eval_pred.predictions) and
            dict format (eval_pred["predictions"]) for compatibility with
            different Trainer versions.

    Returns:
        Dictionary containing the following metrics (all as float):
            - accuracy: Overall classification accuracy (0.0-1.0)
            - precision: Precision for positive class (fake news)
            - recall: Recall for positive class (fake news)
            - f1: F1-score for positive class (harmonic mean of precision/recall)
            - roc_auc: Area under ROC curve (NaN if only one class present)

    Note:
        - Metrics are computed for binary classification (class 0=real, 1=fake)
        - Uses "binary" averaging for precision/recall/F1 (focuses on positive class)
        - ROC AUC uses probability of positive class (probs[:, 1])
        - Returns NaN for ROC AUC if evaluation fails (e.g., single class)
        - Zero division in precision/recall handled by setting to 0.0

    Examples:
        Perfect predictions:
        >>> from types import SimpleNamespace
        >>> logits = np.array([[10, 0], [0, 10], [10, 0]])
        >>> labels = np.array([0, 1, 0])
        >>> eval_pred = SimpleNamespace(predictions=logits, label_ids=labels)
        >>> metrics = compute_metrics(eval_pred)
        >>> print(metrics['accuracy'])  # 1.0
        1.0

        With dict format:
        >>> eval_pred = {"predictions": logits, "label_ids": labels}
        >>> metrics = compute_metrics(eval_pred)
    """
    # Handle both dict format and object format
    if isinstance(eval_pred, dict):
        logits, labels = eval_pred["predictions"], eval_pred["label_ids"]
    else:
        # Assume object with .predictions and .label_ids attributes
        logits, labels = eval_pred.predictions, eval_pred.label_ids

    probs = _softmax(np.array(logits))
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

    # ROC AUC (only if both classes present)
    roc_auc = None
    try:
        roc_auc = float(roc_auc_score(labels, probs[:, 1]))
    except Exception:
        roc_auc = float("nan")

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
    }
