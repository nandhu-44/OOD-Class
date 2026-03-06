from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_id_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_ood_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_cifar10_loaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, datasets.CIFAR10]:
    train_dataset = datasets.CIFAR10(
        root=str(data_root),
        train=True,
        transform=get_id_transforms(train=True),
        download=True,
    )
    train_eval_dataset = datasets.CIFAR10(
        root=str(data_root),
        train=True,
        transform=get_id_transforms(train=False),
        download=True,
    )
    test_dataset = datasets.CIFAR10(
        root=str(data_root),
        train=False,
        transform=get_id_transforms(train=False),
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader, train_eval_dataset


def get_train_eval_loader(
    train_eval_dataset: datasets.CIFAR10,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        train_eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_svhn_loader(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    split: str = "test",
    pin_memory: bool = True,
) -> DataLoader:
    svhn_dataset = datasets.SVHN(
        root=str(data_root),
        split=split,
        transform=get_ood_transforms(),
        download=True,
    )
    return DataLoader(
        svhn_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def download_all_datasets(data_root: Path) -> None:
    datasets.CIFAR10(root=str(data_root), train=True, download=True)
    datasets.CIFAR10(root=str(data_root), train=False, download=True)
    datasets.SVHN(root=str(data_root), split="train", download=True)
    datasets.SVHN(root=str(data_root), split="test", download=True)


def unpack_batch(batch):
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return batch[0], batch[1]
    raise ValueError("Unexpected batch format.")
