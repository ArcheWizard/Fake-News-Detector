from __future__ import annotations

import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    if isinstance(eval_pred, dict):  # safeguard for different Trainer versions
        logits, labels = eval_pred["predictions"], eval_pred["label_ids"]

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
