import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import tqdm

from src.data import (
    get_cifar10_loaders,
    get_svhn_loader,
    get_train_eval_loader,
    unpack_batch,
)
from src.model import CIFARResNet18
from src.ood.energy import energy_confidence_score
from src.ood.mahalanobis import (
    fit_mahalanobis_stats,
    mahalanobis_confidence_score,
    stats_to_numpy_payload,
)
from src.ood.msp import msp_score
from src.utils import ensure_dir, get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Compute OOD scores for CIFAR-10 vs SVHN")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model-path", type=Path, default=Path("outputs/models/resnet18_cifar10.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scores"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--svhn-split", type=str, default="test", choices=["train", "test", "extra"])
    return parser.parse_args()


@torch.no_grad()
def collect_msp_and_energy_scores(model, loader, device, temperature: float):
    model.eval()
    msp_scores = []
    energy_scores = []

    for batch in tqdm(loader, leave=False):
        images, _ = unpack_batch(batch)
        images = images.to(device)

        logits = model(images)
        msp_scores.append(msp_score(logits).cpu().numpy())
        energy_scores.append(energy_confidence_score(logits, temperature=temperature).cpu().numpy())

    return np.concatenate(msp_scores), np.concatenate(energy_scores)


def save_score_file(path: Path, id_scores: np.ndarray, ood_scores: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, id=id_scores, ood=ood_scores)


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = get_device()
    train_loader, id_test_loader, train_eval_dataset = get_cifar10_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    _ = train_loader
    train_eval_loader = get_train_eval_loader(
        train_eval_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    ood_loader = get_svhn_loader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split=args.svhn_split,
        pin_memory=(device.type == "cuda"),
    )

    model = CIFARResNet18(num_classes=10).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    id_msp, id_energy = collect_msp_and_energy_scores(
        model, id_test_loader, device, temperature=args.temperature
    )
    ood_msp, ood_energy = collect_msp_and_energy_scores(
        model, ood_loader, device, temperature=args.temperature
    )

    save_score_file(args.output_dir / "msp_scores.npz", id_msp, ood_msp)
    save_score_file(args.output_dir / "energy_scores.npz", id_energy, ood_energy)

    mahal_stats = fit_mahalanobis_stats(
        model=model,
        loader=train_eval_loader,
        num_classes=10,
        device=device,
        reg_eps=1e-4,
    )
    np.savez(args.output_dir / "mahalanobis_stats.npz", **stats_to_numpy_payload(mahal_stats))

    id_mahal = mahalanobis_confidence_score(model, id_test_loader, mahal_stats, device)
    ood_mahal = mahalanobis_confidence_score(model, ood_loader, mahal_stats, device)
    save_score_file(args.output_dir / "mahalanobis_scores.npz", id_mahal, ood_mahal)

    print("Saved OOD score files:")
    print(f"- {args.output_dir / 'msp_scores.npz'}")
    print(f"- {args.output_dir / 'energy_scores.npz'}")
    print(f"- {args.output_dir / 'mahalanobis_scores.npz'}")


if __name__ == "__main__":
    main()
