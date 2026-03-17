#!/usr/bin/env python3
"""
POS Tagging Model Training Script

A modular training pipeline for transformer and LSTM-based POS tagging models.
Supports automatic OOM fallback, per-tag statistics, and configurable experiments.

Usage:
    python train_pos.py --config path/to/config.json --use-gpu nvidia
    python train_pos.py --sentences 5000 --epochs 10
"""
import argparse
import json
import sys
from pathlib import Path

# Add src to path so imports work from scripts
sys.path.insert(0, str(Path(__file__).parent))

from data import DatasetCache
from training import (
    configure_runtime,
    load_configs,
    make_json_safe,
    print_config_summary,
    set_seed,
    train_one_config,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train POS tagging models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--use-gpu",
        choices=["amd", "nvidia", "none"],
        default="none",
        help="GPU backend to use",
    )
    parser.add_argument(
        "--sentences",
        type=int,
        default=2000,
        help="Number of sentences to use",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=30,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config or config list",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="resources/results/training_results.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="resources/models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--save-models",
        type=bool,
        default=True,
        help="Whether to save trained model files",
    )

    return parser.parse_args()


def run_training(args: argparse.Namespace) -> list[dict]:
    """Run the training pipeline."""
    # Initialize
    set_seed(args.seed)
    configure_runtime(args.use_gpu)

    # Load configs
    configs = load_configs(args.config, args.max_len, args.sentences, args.seed)
    print_config_summary(configs)

    # Train models
    results = []
    dataset_cache = DatasetCache()
    models_dir = Path(args.models_dir)

    for cfg in configs:
        result = train_one_config(
            cfg,
            args.use_gpu,
            models_dir,
            dataset_cache,
            args.save_models,
        )
        results.append(result)

    return results


def save_results(results: list[dict], output_path: str) -> None:
    """Save training results to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(results), f, indent=2)

    print(f"\nSaved results to: {path}")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    results = run_training(args)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
