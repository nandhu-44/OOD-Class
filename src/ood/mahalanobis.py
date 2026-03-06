from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import unpack_batch


@dataclass
class MahalanobisStats:
    class_means: torch.Tensor
    precision: torch.Tensor


@torch.no_grad()
def extract_features_and_labels(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_features = []
    all_labels = []

    for batch in loader:
        images, labels = unpack_batch(batch)
        images = images.to(device)
        labels = labels.to(device)

        _, features = model.forward_with_features(images)
        all_features.append(features)
        all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return features, labels


def fit_mahalanobis_stats(
    model: torch.nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    reg_eps: float = 1e-4,
) -> MahalanobisStats:
    features, labels = extract_features_and_labels(model, loader, device)
    feat_dim = features.shape[1]

    class_means = []
    centered = []

    for c in range(num_classes):
        class_feats = features[labels == c]
        mean_c = class_feats.mean(dim=0)
        class_means.append(mean_c)
        centered.append(class_feats - mean_c)

    class_means = torch.stack(class_means, dim=0)
    centered_all = torch.cat(centered, dim=0)

    cov = (centered_all.T @ centered_all) / max(centered_all.shape[0] - 1, 1)
    cov += reg_eps * torch.eye(feat_dim, device=cov.device)
    precision = torch.linalg.inv(cov)

    return MahalanobisStats(class_means=class_means, precision=precision)


@torch.no_grad()
def mahalanobis_confidence_score(
    model: torch.nn.Module,
    loader: DataLoader,
    stats: MahalanobisStats,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    scores = []

    means = stats.class_means.to(device)
    precision = stats.precision.to(device)

    for batch in loader:
        images, _ = unpack_batch(batch)
        images = images.to(device)

        _, features = model.forward_with_features(images)
        diff = features.unsqueeze(1) - means.unsqueeze(0)

        left = torch.matmul(diff, precision)
        d2 = (left * diff).sum(dim=2)
        min_d2, _ = d2.min(dim=1)

        # Higher score should indicate ID-like samples.
        scores.append((-min_d2).detach().cpu().numpy())

    return np.concatenate(scores, axis=0)


def stats_to_numpy_payload(stats: MahalanobisStats) -> Dict[str, np.ndarray]:
    return {
        "class_means": stats.class_means.detach().cpu().numpy(),
        "precision": stats.precision.detach().cpu().numpy(),
    }


def stats_from_numpy_payload(payload: Dict[str, np.ndarray]) -> MahalanobisStats:
    return MahalanobisStats(
        class_means=torch.from_numpy(payload["class_means"]).float(),
        precision=torch.from_numpy(payload["precision"]).float(),
    )
