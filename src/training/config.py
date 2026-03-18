"""Configuration loading and defaults for training."""
import json
import re
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
    configs = _expand_sentence_ranges(configs)
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

        if isinstance(cfg["sentences"], bool) or not isinstance(cfg["sentences"], int):
            raise ValueError(
                f"Config '{cfg.get('name', '<unnamed>')}' has invalid 'sentences' value. "
                "Only positive integers are supported."
            )
        if cfg["sentences"] <= 0:
            raise ValueError(
                f"Config '{cfg.get('name', '<unnamed>')}' has invalid 'sentences' value. "
                "'sentences' must be > 0."
            )


def _expand_sentence_ranges(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand configs that define sentence ranges into multiple concrete configs.

    Supported keys:
      - sentence_start: int
      - sentence_end: int
      - sentence_step: int

    Name behavior:
      - If name includes "{sentences}", it is replaced with the sentence label.
      - Else, a trailing "_s<number|max>" is replaced.
      - Else, "_s<label>" is appended.
    """
    expanded: list[dict[str, Any]] = []

    for cfg in configs:
        if cfg.get("sentences") == "max":
            raise ValueError(
                f"Config '{cfg.get('name', '<unnamed>')}' uses sentences='max', "
                "which is no longer supported. Use an explicit integer sentence count."
            )
        if "sentence_include_max" in cfg:
            raise ValueError(
                f"Config '{cfg.get('name', '<unnamed>')}' uses 'sentence_include_max', "
                "which is no longer supported. Use an explicit numeric range."
            )

        has_range = all(k in cfg for k in ("sentence_start", "sentence_end", "sentence_step"))
        if not has_range:
            expanded.append(cfg)
            continue

        start = int(cfg["sentence_start"])
        end = int(cfg["sentence_end"])
        step = int(cfg["sentence_step"])

        if start <= 0 or end <= 0 or step <= 0:
            raise ValueError(
                "sentence_start, sentence_end and sentence_step must be positive integers"
            )
        if start > end:
            raise ValueError("sentence_start must be <= sentence_end")

        sentence_values: list[int | str] = list(range(start, end + 1, step))
        if sentence_values[-1] != end:
            sentence_values.append(end)

        for sent_value in sentence_values:
            concrete = dict(cfg)
            concrete["sentences"] = sent_value
            concrete["name"] = _name_for_sentence(cfg.get("name", "run"), sent_value)

            # Remove range-only keys from concrete runs.
            concrete.pop("sentence_start", None)
            concrete.pop("sentence_end", None)
            concrete.pop("sentence_step", None)

            expanded.append(concrete)

    return expanded


def _name_for_sentence(base_name: str, sentence_value: int | str) -> str:
    """Create a run name that includes the sentence value label."""
    label = str(sentence_value)

    if "{sentences}" in base_name:
        return base_name.replace("{sentences}", label)

    # Replace existing sentence suffix if present.
    replaced = re.sub(r"_s\d+$", f"_s{label}", base_name)
    if replaced != base_name:
        return replaced

    return f"{base_name}_s{label}"


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

    unique_sentences = sorted({int(cfg["sentences"]) for cfg in configs})
    unique_maxlens = sorted({int(cfg["maxlen"]) for cfg in configs})
    unique_split_seeds = sorted({int(cfg["split_seed"]) for cfg in configs})
    unique_run_seeds = sorted({int(cfg["seed"]) for cfg in configs})

    print(f"Sentence-count values in sweep: {unique_sentences}")
    print(f"Max-length values in sweep: {unique_maxlens}")
    print(f"Train/test split seeds in sweep: {unique_split_seeds}")
    print(f"Run seeds in sweep: {unique_run_seeds}")
