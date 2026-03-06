import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src.data import get_cifar10_loaders, unpack_batch
from src.model import CIFARResNet18
from src.utils import ensure_dir, get_device, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 classifier")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/models"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for batch in loader:
        images, labels = unpack_batch(batch)
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / max(total, 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = get_device()
    train_loader, test_loader, _ = get_cifar10_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = CIFARResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_test_acc = -1.0
    best_epoch = -1
    best_model_path = args.output_dir / "resnet18_cifar10_best.pt"

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for batch in loop:
            images, labels = unpack_batch(batch)
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            loop.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_loader.dataset)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, test_acc={test_acc:.4f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "test_accuracy": best_test_acc,
                    "epoch": best_epoch,
                    "seed": args.seed,
                },
                best_model_path,
            )

    final_test_acc = evaluate(model, test_loader, device)
    print(f"Final CIFAR-10 test accuracy: {final_test_acc:.4f}")

    model_path = args.output_dir / "resnet18_cifar10.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "test_accuracy": final_test_acc,
            "epoch": args.epochs,
            "seed": args.seed,
        },
        model_path,
    )

    save_json(
        args.output_dir / "training_summary.json",
        {
            "model_path": str(model_path),
            "best_model_path": str(best_model_path),
            "test_accuracy": final_test_acc,
            "best_test_accuracy": best_test_acc,
            "best_epoch": best_epoch,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()
