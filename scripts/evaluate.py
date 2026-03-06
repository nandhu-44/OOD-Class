import argparse
import csv
from pathlib import Path

import numpy as np

from src.ood.metrics import evaluate_scores
from src.ood.plots import plot_comparison_bars, plot_roc_curves, plot_score_histograms
from src.utils import ensure_dir, save_json


METHOD_FILES = {
    "MSP": "msp_scores.npz",
    "Energy": "energy_scores.npz",
    "Mahalanobis": "mahalanobis_scores.npz",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate OOD detection metrics and plots")
    parser.add_argument("--score-dir", type=Path, default=Path("outputs/scores"))
    parser.add_argument("--result-dir", type=Path, default=Path("outputs/results"))
    return parser.parse_args()


def load_score_payload(path: Path):
    payload = np.load(path)
    return {
        "id": payload["id"],
        "ood": payload["ood"],
    }


def main():
    args = parse_args()
    ensure_dir(args.result_dir)

    scores = {}
    metrics_table = {}

    for method, filename in METHOD_FILES.items():
        path = args.score_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing score file for {method}: {path}")

        payload = load_score_payload(path)
        scores[method] = payload
        metrics_table[method] = evaluate_scores(payload["id"], payload["ood"])

    metrics_csv_path = args.result_dir / "metrics_table.csv"
    with metrics_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "AUROC", "FPR95", "DetectionAccuracy", "BestThreshold"])
        for method, row in metrics_table.items():
            writer.writerow(
                [
                    method,
                    f"{row['AUROC']:.6f}",
                    f"{row['FPR95']:.6f}",
                    f"{row['DetectionAccuracy']:.6f}",
                    f"{row['BestThreshold']:.6f}",
                ]
            )

    save_json(args.result_dir / "metrics_table.json", metrics_table)

    plot_roc_curves(scores, args.result_dir / "roc_curves.png")
    plot_score_histograms(scores, args.result_dir / "histograms")
    plot_comparison_bars(metrics_table, args.result_dir / "comparison_bars.png")

    print("OOD Evaluation Summary")
    for method, row in metrics_table.items():
        print(
            f"{method:12s} | AUROC={row['AUROC']:.4f} | "
            f"FPR95={row['FPR95']:.4f} | DetectionAcc={row['DetectionAccuracy']:.4f}"
        )

    print(f"Saved metrics table: {metrics_csv_path}")
    print(f"Saved plots under: {args.result_dir}")


if __name__ == "__main__":
    main()
