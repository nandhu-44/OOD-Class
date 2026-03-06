from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def plot_roc_curves(scores: Dict[str, Dict[str, np.ndarray]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))

    for method, payload in scores.items():
        id_scores = payload["id"]
        ood_scores = payload["ood"]
        y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
        y_score = np.concatenate([id_scores, ood_scores])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, label=method)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: CIFAR-10 (ID) vs SVHN (OOD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_score_histograms(scores: Dict[str, Dict[str, np.ndarray]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for method, payload in scores.items():
        id_scores = payload["id"]
        ood_scores = payload["ood"]

        plt.figure(figsize=(8, 5))
        plt.hist(id_scores, bins=50, alpha=0.6, density=True, label="ID (CIFAR-10)")
        plt.hist(ood_scores, bins=50, alpha=0.6, density=True, label="OOD (SVHN)")
        plt.xlabel("Score")
        plt.ylabel("Density")
        plt.title(f"Score Distribution: {method}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"hist_{method.lower()}.png", dpi=200)
        plt.close()


def plot_comparison_bars(metrics_table, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    methods = list(metrics_table.keys())
    auroc = [metrics_table[m]["AUROC"] for m in methods]
    fpr95 = [metrics_table[m]["FPR95"] for m in methods]
    det_acc = [metrics_table[m]["DetectionAccuracy"] for m in methods]

    x = np.arange(len(methods))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, auroc, width=width, label="AUROC")
    plt.bar(x, [1.0 - v for v in fpr95], width=width, label="1 - FPR95")
    plt.bar(x + width, det_acc, width=width, label="Detection Accuracy")

    plt.xticks(x, methods)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Value")
    plt.title("OOD Detection Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
