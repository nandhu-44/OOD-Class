import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import download_all_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Download CIFAR-10 and SVHN datasets")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    download_all_datasets(args.data_root)
    print(f"Datasets downloaded under: {args.data_root.resolve()}")


if __name__ == "__main__":
    main()
