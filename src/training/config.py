"""Configuration loading and defaults for training."""
import json
from typing import Any


def load_configs(
    config_path: str | None,
    max_len: int,
    sentences: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Load configurations from JSON file or use defaults."""
    if config_path is None:
        return _default_configs(max_len, sentences, seed)

    configs = _load_from_file(config_path)
    _apply_defaults(configs, max_len, sentences, seed)
    return configs


def _load_from_file(config_path: str) -> list[dict[str, Any]]:
    """Load and parse configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data

    raise ValueError("Config JSON must be either an object or a list of objects.")


def _apply_defaults(configs: list[dict[str, Any]], max_len: int, sentences: int, seed: int) -> None:
    """Apply default values to configurations missing required fields."""
    for cfg in configs:
        cfg.setdefault("sentences", sentences)
        cfg.setdefault("maxlen", max_len)
        cfg.setdefault("split_seed", seed)
        cfg.setdefault("seed", seed)


def _default_configs(max_len: int, sentences: int, seed: int) -> list[dict[str, Any]]:
    """Return default training configurations."""
    return [
        {
            "name": "transformer_small",
            "model_type": "transformer",
            "sentences": sentences,
            "maxlen": max_len,
            "split_seed": seed,
            "seed": seed,
            "embed_dim": 128,
            "num_heads": 4,
            "ff_dim": 128,
            "num_layers": 1,
            "dropout": 0.1,
            "lr": 0.001,
            "epochs": 5,
            "batch_size": 32,
        },
        {
            "name": "lstm_baseline",
            "model_type": "lstm",
            "sentences": sentences,
            "maxlen": max_len,
            "split_seed": seed,
            "seed": seed,
            "embed_dim": 128,
            "lstm_units": 64,
            "lr": 0.001,
            "epochs": 10,
            "batch_size": 32,
        },
    ]


def print_config_summary(configs: list[dict[str, Any]]) -> None:
    """Print a summary of loaded configurations."""
    print(f"\nLoaded {len(configs)} config(s).")

    unique_sentences = sorted(
        {cfg["sentences"] for cfg in configs},
        key=lambda x: float("inf") if x == "max" else x,
    )
    unique_maxlens = sorted({int(cfg["maxlen"]) for cfg in configs})
    unique_split_seeds = sorted({int(cfg["split_seed"]) for cfg in configs})
    unique_run_seeds = sorted({int(cfg["seed"]) for cfg in configs})

    print(f"Sentence-count values in sweep: {unique_sentences}")
    print(f"Max-length values in sweep: {unique_maxlens}")
    print(f"Train/test split seeds in sweep: {unique_split_seeds}")
    print(f"Run seeds in sweep: {unique_run_seeds}")
