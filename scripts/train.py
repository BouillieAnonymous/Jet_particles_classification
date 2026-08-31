"""Run one reproducible jet-classification experiment from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.experiment import ExperimentConfig, run_experiment
from src.features import FEATURE_SETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("fixed", "dynamic"), default="dynamic"
    )
    parser.add_argument(
        "--features", choices=tuple(FEATURE_SETS), default="full"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument(
        "--max-epochs", "--epochs", dest="max_epochs", type=int, default=50
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-entries", type=int)
    parser.add_argument("--root-path", type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_experiment(
        ExperimentConfig(
            repository_root=REPOSITORY_ROOT,
            root_path=args.root_path,
            results_dir=args.results_dir,
            model=args.model,
            features=args.features,
            seed=args.seed,
            split_seed=args.split_seed,
            k=args.k,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_entries=args.max_entries,
            device=args.device,
        )
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
