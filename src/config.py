from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    data_root: Path = Path("data")
    output_root: Path = Path("outputs")
    model_dir: Path = Path("outputs/models")
    score_dir: Path = Path("outputs/scores")
    result_dir: Path = Path("outputs/results")
    batch_size: int = 128
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 30
    seed: int = 42
    temperature: float = 1.0
