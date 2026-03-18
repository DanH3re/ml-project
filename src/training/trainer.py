"""Training logic with OOM fallback handling."""
import json
import time
from pathlib import Path
from typing import Any

import tensorflow as tf
from tensorflow import keras

from data import DatasetCache, prepare_split_for_config
from evaluation import evaluate_model, print_tag_statistics

from .models import build_model
from .utils import make_json_safe, set_seed, slugify


# Default batch sizes to try when OOM occurs
FALLBACK_BATCH_SIZES = [32, 16, 8, 4, 2, 1]


def train_one_config(
    config: dict[str, Any],
    use_gpu: str,
    models_dir: Path,
    dataset_cache: DatasetCache,
    run_seed: int,
    run_index: int = 1,
    run_count: int = 1,
    save_models: bool = True,
) -> dict[str, Any]:
    """Train a single model configuration with OOM fallback handling."""
    name = config.get("name", f"{config['model_type']}_run")
    run_seed = int(run_seed)
    set_seed(run_seed)

    config = dict(config)
    config.pop("group", None)
    config["seed"] = run_seed

    # Prepare data
    prepared = prepare_split_for_config(config, dataset_cache)
    config = _enrich_config(config, prepared)

    _print_run_header(name, config, prepared)

    # Train model
    model, history, train_info = _train_with_fallback(
        config, use_gpu, prepared["X_train"], prepared["y_train"]
    )

    # Evaluate model
    metrics = evaluate_model(
        model,
        prepared["X_test"],
        prepared["y_test"],
        vocab=prepared["vocab"],
        predict_batch_size=int(config.get("predict_batch_size", 32)),
        model_type=config["model_type"],
        config=config,
        use_gpu=use_gpu,
    )

    # Save model
    model_path = _save_model(model, name, models_dir, save_models)

    # Build result
    result = _build_result(
        name,
        config,
        prepared,
        model,
        history,
        metrics,
        model_path,
        train_info,
        run_seed,
        run_index,
        run_count,
        save_models,
    )

    _print_results(result, model_path, save_models)

    return result


def _enrich_config(config: dict[str, Any], prepared: dict[str, Any]) -> dict[str, Any]:
    """Add dataset-derived values to config."""
    config = dict(config)
    config["vocab_size"] = prepared["vocab_size"]
    config["num_tags"] = prepared["num_tags"]
    return config


def _print_run_header(name: str, config: dict[str, Any], prepared: dict[str, Any]) -> None:
    """Print training run header."""
    print("\n" + "=" * 80)
    print(f"Training run: {name}")
    print(json.dumps(make_json_safe(config), indent=2))
    print("Dataset shapes:")
    print(json.dumps(make_json_safe(prepared["dataset_shape"]), indent=2))
    print("=" * 80)


def _train_with_fallback(
    config: dict[str, Any],
    use_gpu: str,
    X_train,
    y_train,
) -> tuple[keras.Model, keras.callbacks.History, dict[str, Any]]:
    """Train model with OOM fallback to smaller batches and CPU."""
    batch_sizes = _get_batch_sizes_to_try(config.get("batch_size", 32))
    callbacks = _create_callbacks(config)

    # Try default device
    result = _try_train_on_device(
        config, use_gpu, X_train, y_train, batch_sizes, callbacks, device="default"
    )
    if result is not None:
        return result

    # Try CPU fallback
    print("\n" + "=" * 80)
    print("Attempting CPU fallback for training...")
    print("=" * 80)

    result = _try_train_on_device(
        config, "none", X_train, y_train, batch_sizes, callbacks, device="cpu"
    )
    if result is not None:
        return result

    raise RuntimeError("Training failed on both default device and CPU.")


def _try_train_on_device(
    config: dict[str, Any],
    use_gpu: str,
    X_train,
    y_train,
    batch_sizes: list[int],
    callbacks: list[keras.callbacks.Callback],
    device: str,
) -> tuple[keras.Model, keras.callbacks.History, dict[str, Any]] | None:
    """Try training with progressively smaller batch sizes on a device."""
    original_batch_size = batch_sizes[0]
    fit_verbose = _resolve_fit_verbose(config)

    for bs in batch_sizes:
        try:
            context = tf.device("/CPU:0") if device == "cpu" else _null_context()

            with context:
                print(f"Training on {device} with batch_size={bs}")
                keras.backend.clear_session()
                model = build_model(config["model_type"], config, use_gpu)

                if bs == original_batch_size:
                    model.summary()

                start = time.perf_counter()
                history = model.fit(
                    X_train,
                    y_train,
                    batch_size=bs,
                    epochs=config["epochs"],
                    validation_split=config.get("validation_split", 0.2),
                    verbose=fit_verbose,
                    callbacks=callbacks,
                )
                train_time_sec = time.perf_counter() - start

            print(f"Training succeeded with batch_size={bs}")

            train_info = {
                "train_time_sec": train_time_sec,
                "train_batch_size_used": bs,
                "train_device_used": device if device == "cpu" else "default",
            }

            return model, history, train_info

        except tf.errors.ResourceExhaustedError:
            print(f"OOM on {device} with batch_size={bs}. Trying smaller batch...")
            keras.backend.clear_session()

    return None


def _resolve_fit_verbose(config: dict[str, Any]) -> int:
    """Resolve Keras fit verbosity (0=silent, 1=progress bar, 2=one line/epoch)."""
    raw = config.get("fit_verbose", 2)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(
            f"Config '{config.get('name', '<unnamed>')}' has invalid fit_verbose={raw}. "
            "Expected integer 0, 1, or 2."
        )
    if raw not in {0, 1, 2}:
        raise ValueError(
            f"Config '{config.get('name', '<unnamed>')}' has invalid fit_verbose={raw}. "
            "Expected integer 0, 1, or 2."
        )
    return int(raw)


def _get_batch_sizes_to_try(initial_batch_size: int) -> list[int]:
    """Get ordered list of batch sizes to try."""
    batch_sizes = [initial_batch_size]
    for bs in FALLBACK_BATCH_SIZES:
        if bs not in batch_sizes:
            batch_sizes.append(bs)
    return batch_sizes


def _create_callbacks(config: dict[str, Any]) -> list[keras.callbacks.Callback]:
    """Create training callbacks based on config."""
    callbacks = []

    if config.get("early_stopping", True):
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.get("patience", 3),
                restore_best_weights=True,
            )
        )

    return callbacks


def _save_model(model: keras.Model, name: str, models_dir: Path, save_models: bool) -> Path:
    """Save model to disk if enabled."""
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{slugify(name)}.keras"

    if save_models:
        model.save(model_path)

    return model_path


def _build_result(
    name: str,
    config: dict[str, Any],
    prepared: dict[str, Any],
    model: keras.Model,
    history: keras.callbacks.History,
    metrics: dict[str, Any],
    model_path: Path,
    train_info: dict[str, Any],
    run_seed: int,
    run_index: int,
    run_count: int,
    save_models: bool,
) -> dict[str, Any]:
    """Build the result dictionary for a training run."""
    dataset_meta = {
        "dataset": config.get("dataset", "brown"),
        "sentences": int(prepared["actual_sentences"]),
        "sentences_config": config["sentences"],
        "maxlen": int(config["maxlen"]),
        "split_seed": int(prepared.get("global_split_seed", 42)),
        "seed": int(run_seed),
        "vocab_size": int(prepared["vocab_size"]),
        "num_tags": int(prepared["num_tags"]),
        "shapes": make_json_safe(prepared["dataset_shape"]),
    }
    if "global_split_path" in prepared:
        dataset_meta["global_split_path"] = str(prepared["global_split_path"])
    if "global_split_train_pool_size" in prepared:
        dataset_meta["global_split_train_pool_size"] = int(prepared["global_split_train_pool_size"])
    if "global_split_test_size" in prepared:
        dataset_meta["global_split_test_size"] = int(prepared["global_split_test_size"])
    if "global_test_size" in prepared:
        dataset_meta["global_test_size"] = float(prepared["global_test_size"])

    return {
        "name": name,
        "model_type": config["model_type"],
        "config": make_json_safe(config),
        "dataset_meta": dataset_meta,
        "num_params": int(model.count_params()),
        "run_seed": int(run_seed),
        "run_index": int(run_index),
        "run_count": int(run_count),
        "train_time_sec": float(train_info["train_time_sec"]),
        "train_batch_size_used": int(train_info["train_batch_size_used"]),
        "train_device_used": str(train_info["train_device_used"]),
        "best_val_accuracy": float(max(history.history.get("val_accuracy", [0.0]))),
        "best_val_loss": float(min(history.history.get("val_loss", [float("inf")]))),
        "epochs_ran": int(len(history.history.get("loss", []))),
        "history": make_json_safe(history.history),
        "test_metrics": metrics,
        "model_path": str(model_path),
        "model_saved": bool(save_models),
    }


def _print_results(result: dict[str, Any], model_path: Path, save_models: bool) -> None:
    """Print training results summary."""
    print("\nTest metrics:")
    print(json.dumps({
        k: v for k, v in result["test_metrics"].items()
        if k not in ("per_tag", "tag_counts")  # Don't print full per-tag stats
    }, indent=2))

    # Print tag statistics summary
    print_tag_statistics(result["test_metrics"])

    if save_models:
        print(f"\nSaved model to: {model_path}")
    else:
        print(f"\nModel saving disabled. Would save to: {model_path}")


class _null_context:
    """No-op context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
