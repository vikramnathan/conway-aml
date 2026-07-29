"""Binary classification metrics (no sklearn dependency).

Operate on 1-D numpy arrays of scores (probabilities) and 0/1 labels. Used to
evaluate the AML head on held-out test accounts. Reports the paper's AML metric
(F0.5) alongside ROC-AUC, PR-AUC (average precision), and best-threshold F1.
"""

from __future__ import annotations

import numpy as np


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via the rank (Mann–Whitney U) formulation; handles ties."""
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    s = scores[order]
    ranks_sorted = np.arange(1, len(scores) + 1, dtype=float)
    # average ranks within tie groups
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks_sorted[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranks[order] = ranks_sorted
    sum_pos = ranks[labels].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (area under precision-recall via the AP sum)."""
    labels = labels.astype(int)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / labels.sum()
    # AP = sum over thresholds of (recall_i - recall_{i-1}) * precision_i
    prev_recall = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - prev_recall) * precision))


def best_fbeta(scores: np.ndarray, labels: np.ndarray, beta: float) -> tuple[float, float]:
    """Max F-beta over all thresholds. Returns (best_fbeta, threshold)."""
    labels = labels.astype(int)
    P = labels.sum()
    if P == 0:
        return float("nan"), 0.5
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    s = scores[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / P
    b2 = beta * beta
    denom = b2 * precision + recall
    fbeta = np.where(denom > 0, (1 + b2) * precision * recall / np.maximum(denom, 1e-12), 0.0)
    k = int(np.argmax(fbeta))
    return float(fbeta[k]), float(s[k])


def all_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    f1, thr1 = best_fbeta(scores, labels, 1.0)
    f05, thr05 = best_fbeta(scores, labels, 0.5)
    return {
        "roc_auc": roc_auc(scores, labels),
        "pr_auc": pr_auc(scores, labels),
        "f1": f1, "f1_threshold": thr1,
        "f0.5": f05, "f0.5_threshold": thr05,
        "n": int(len(labels)), "n_pos": int(labels.sum()),
        "pos_rate": float(labels.mean()) if len(labels) else float("nan"),
    }
