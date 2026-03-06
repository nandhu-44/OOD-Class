from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_fpr95(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])


def compute_detection_accuracy(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    detection_acc = 0.5 * (tpr + (1.0 - fpr))
    best_idx = int(np.argmax(detection_acc))
    return float(detection_acc[best_idx]), float(thresholds[best_idx])


def evaluate_scores(id_scores: np.ndarray, ood_scores: np.ndarray) -> Dict[str, float]:
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])

    auroc = float(roc_auc_score(y_true, y_score))
    fpr95 = compute_fpr95(y_true, y_score)
    det_acc, best_threshold = compute_detection_accuracy(y_true, y_score)

    return {
        "AUROC": auroc,
        "FPR95": fpr95,
        "DetectionAccuracy": det_acc,
        "BestThreshold": best_threshold,
    }
