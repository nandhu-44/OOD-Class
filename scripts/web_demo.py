import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import CIFAR10_MEAN, CIFAR10_STD, get_cifar10_loaders, get_train_eval_loader
from src.model import CIFARResNet18
from src.ood.energy import energy_confidence_score
from src.ood.mahalanobis import (
    MahalanobisStats,
    fit_mahalanobis_stats,
    stats_from_numpy_payload,
)
from src.ood.metrics import evaluate_scores
from src.ood.msp import msp_score
from src.utils import get_device, set_seed


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

METHOD_FILES = {
    "MSP": "msp_scores.npz",
    "Energy": "energy_scores.npz",
    "Mahalanobis": "mahalanobis_scores.npz",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive web demo for CIFAR-10 + OOD scores")
    parser.add_argument("--model-path", type=Path, default=Path("outputs/models/resnet18_cifar10_best.pt"))
    parser.add_argument("--mahal-stats-path", type=Path, default=Path("outputs/scores/mahalanobis_stats.npz"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--score-dir", type=Path, default=Path("outputs/scores"))
    parser.add_argument(
        "--decision-method",
        type=str,
        default="MSP",
        choices=["MSP", "Energy", "Mahalanobis"],
        help="Primary method used for overall ID/OOD decision when thresholds are available.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio link.")
    return parser.parse_args()


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            "Pass --model-path with a valid .pt file."
        )
    model = CIFARResNet18(num_classes=10).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_calibration_thresholds(score_dir: Path) -> Dict[str, Optional[float]]:
    thresholds: Dict[str, Optional[float]] = {k: None for k in METHOD_FILES}
    for method, filename in METHOD_FILES.items():
        score_path = score_dir / filename
        if not score_path.exists():
            continue

        payload = np.load(score_path)
        id_scores = payload["id"]
        ood_scores = payload["ood"]
        metrics = evaluate_scores(id_scores, ood_scores)
        thresholds[method] = float(metrics["BestThreshold"])
    return thresholds


def maybe_load_or_fit_mahalanobis(
    model: torch.nn.Module,
    mahal_stats_path: Path,
    data_root: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> MahalanobisStats:
    if mahal_stats_path.exists():
        payload = np.load(mahal_stats_path)
        return stats_from_numpy_payload({"class_means": payload["class_means"], "precision": payload["precision"]})

    # Fallback: fit stats from CIFAR-10 train split if precomputed stats are missing.
    _, _, train_eval_dataset = get_cifar10_loaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    train_eval_loader = get_train_eval_loader(
        train_eval_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    stats = fit_mahalanobis_stats(
        model=model,
        loader=train_eval_loader,
        num_classes=10,
        device=device,
    )

    mahal_stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        mahal_stats_path,
        class_means=stats.class_means.detach().cpu().numpy(),
        precision=stats.precision.detach().cpu().numpy(),
    )
    return stats


@torch.no_grad()
def predict(
    image: Image.Image,
    model: torch.nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    mahal_stats: MahalanobisStats,
    temperature: float,
    thresholds: Dict[str, Optional[float]],
    decision_method: str,
) -> Tuple[str, List[List[str]], List[List[str]], Dict[str, float]]:
    if image is None:
        return "Upload an image first.", [], [], {}

    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    logits, features = model.forward_with_features(x)

    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    top_idx = int(np.argmax(probs))
    top_label = CIFAR10_CLASSES[top_idx]
    top_prob = float(probs[top_idx])

    msp = float(msp_score(logits)[0].item())
    energy = float(energy_confidence_score(logits, temperature=temperature)[0].item())

    means = mahal_stats.class_means.to(device)
    precision = mahal_stats.precision.to(device)
    diff = features.unsqueeze(1) - means.unsqueeze(0)
    d2 = (torch.matmul(diff, precision) * diff).sum(dim=2)
    mahal_conf = float((-d2.min(dim=1).values)[0].item())

    ranked = np.argsort(-probs)
    top_rows = [[CIFAR10_CLASSES[i], f"{probs[i]:.4f}"] for i in ranked[:10]]

    summary = (
        f"Top-1 Prediction: {top_label} ({top_prob:.4f})\n"
        f"OOD scores (higher means more ID-like): MSP={msp:.4f}, Energy={energy:.4f}, Mahalanobis={mahal_conf:.4f}"
    )

    ood_dict = {
        "MSP": msp,
        "Energy": energy,
        "Mahalanobis": mahal_conf,
    }

    ood_rows = []
    for method in ["MSP", "Energy", "Mahalanobis"]:
        score = ood_dict[method]
        thr = thresholds.get(method)
        if thr is None:
            ood_rows.append([method, f"{score:.4f}", "N/A", "Not calibrated"])
        else:
            decision = "ID-like" if score >= thr else "OOD-like"
            ood_rows.append([method, f"{score:.4f}", f"{thr:.4f}", decision])

    chosen_thr = thresholds.get(decision_method)
    if chosen_thr is None:
        summary += f"\nOverall decision ({decision_method}): Not calibrated (run scripts/score_ood.py first)."
    else:
        overall = "ID-like" if ood_dict[decision_method] >= chosen_thr else "OOD-like"
        summary += f"\nOverall decision ({decision_method}): {overall}"

    return summary, top_rows, ood_rows, ood_dict


def build_app(
    model: torch.nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    mahal_stats: MahalanobisStats,
    temperature: float,
    thresholds: Dict[str, Optional[float]],
    decision_method: str,
):
    with gr.Blocks(title="CIFAR-10 OOD Demo") as demo:
        gr.Markdown("# CIFAR-10 Classifier + OOD Detection Demo")
        gr.Markdown(
            "Upload any image to see CIFAR-10 prediction and OOD confidence scores (MSP, Energy, Mahalanobis)."
        )

        with gr.Row():
            inp = gr.Image(type="pil", label="Input Image")
            out_text = gr.Textbox(label="Prediction Summary", lines=3)

        out_table = gr.Dataframe(
            headers=["Class", "Probability"],
            datatype=["str", "str"],
            label="Class Probabilities (sorted)",
            row_count=10,
            col_count=(2, "fixed"),
        )
        ood_table = gr.Dataframe(
            headers=["Method", "Score", "Threshold", "Decision"],
            datatype=["str", "str", "str", "str"],
            label="OOD Decision Table",
            row_count=3,
            col_count=(4, "fixed"),
        )
        out_scores = gr.Label(label="OOD Confidence Scores")

        run_btn = gr.Button("Run Inference")
        run_btn.click(
            fn=lambda image: predict(
                image=image,
                model=model,
                transform=transform,
                device=device,
                mahal_stats=mahal_stats,
                temperature=temperature,
                thresholds=thresholds,
                decision_method=decision_method,
            ),
            inputs=[inp],
            outputs=[out_text, out_table, ood_table, out_scores],
        )

    return demo


def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    model = load_model(args.model_path, device)
    transform = build_transform()
    thresholds = load_calibration_thresholds(args.score_dir)
    mahal_stats = maybe_load_or_fit_mahalanobis(
        model=model,
        mahal_stats_path=args.mahal_stats_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    app = build_app(
        model=model,
        transform=transform,
        device=device,
        mahal_stats=mahal_stats,
        temperature=args.temperature,
        thresholds=thresholds,
        decision_method=args.decision_method,
    )
    print(f"Model loaded from: {args.model_path}")
    print(f"Mahalanobis stats path: {args.mahal_stats_path}")
    print(f"Score calibration dir: {args.score_dir}")
    print(f"Primary decision method: {args.decision_method}")
    app.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
