import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run full OOD experiment pipeline")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()

    model_dir = args.output_root / "models"
    score_dir = args.output_root / "scores"
    result_dir = args.output_root / "results"
    model_path = model_dir / "resnet18_cifar10.pt"

    py = sys.executable

    run_step([py, "scripts/prepare_data.py", "--data-root", str(args.data_root)])
    run_step(
        [
            py,
            "scripts/train.py",
            "--data-root",
            str(args.data_root),
            "--output-dir",
            str(model_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
        ]
    )
    run_step(
        [
            py,
            "scripts/score_ood.py",
            "--data-root",
            str(args.data_root),
            "--model-path",
            str(model_path),
            "--output-dir",
            str(score_dir),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--temperature",
            str(args.temperature),
            "--seed",
            str(args.seed),
        ]
    )
    run_step(
        [
            py,
            "scripts/evaluate.py",
            "--score-dir",
            str(score_dir),
            "--result-dir",
            str(result_dir),
        ]
    )


if __name__ == "__main__":
    main()
