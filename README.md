# OOD Detection Experiment: CIFAR-10 (ID) vs SVHN (OOD)

This project implements a research-style Out-of-Distribution (OOD) experiment using PyTorch.

## Experiment Setup

- In-distribution (ID): CIFAR-10
- Out-of-distribution (OOD): SVHN
- Classifier training data: CIFAR-10 train split only
- ID evaluation data: CIFAR-10 test split
- OOD evaluation data: SVHN split (`test` by default)

## OOD Methods

- MSP (Maximum Softmax Probability)
- Energy-based confidence score
- Mahalanobis confidence score from penultimate features

## Metrics

- AUROC
- FPR95
- Detection Accuracy

## Outputs

- CIFAR-10 classification accuracy: `outputs/models/training_summary.json`
- OOD score files: `outputs/scores/*.npz`
- Metrics table: `outputs/results/metrics_table.csv` and `outputs/results/metrics_table.json`
- Plots:
  - `outputs/results/roc_curves.png`
  - `outputs/results/histograms/*.png`
  - `outputs/results/comparison_bars.png`

## Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

## Quick Start (Full Pipeline)

```bash
python scripts/run_experiment.py --epochs 30 --batch-size 128 --num-workers 4
```

## Step-by-Step

1. Download datasets:

```bash
python scripts/prepare_data.py --data-root data
```

1. Train classifier on CIFAR-10:

```bash
python scripts/train.py --data-root data --output-dir outputs/models --epochs 30
```

1. Score ID vs OOD samples:

```bash
python scripts/score_ood.py --data-root data --model-path outputs/models/resnet18_cifar10.pt --output-dir outputs/scores
```

1. Evaluate and plot:

```bash
python scripts/evaluate.py --score-dir outputs/scores --result-dir outputs/results
```

## Notes

- All dataset loaders use `download=True`, so data is automatically fetched when not already available.
- Reproducibility is controlled by `--seed`.
- For large hardware, consider increasing `--batch-size` and `--num-workers`.
