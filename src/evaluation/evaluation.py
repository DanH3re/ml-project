"""Model evaluation with OOM fallback handling."""
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .metrics import compute_metrics


# Default batch sizes to try when OOM occurs
FALLBACK_BATCH_SIZES = [32, 16, 8, 4, 2, 1]


def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    vocab: Any = None,
    predict_batch_size: int = 32,
    allow_cpu_fallback: bool = True,
    model_type: str | None = None,
    config: dict[str, Any] | None = None,
    use_gpu: str = "none",
) -> dict[str, Any]:
    """
    Evaluate model with automatic OOM fallback.

    Tries progressively smaller batch sizes, then falls back to CPU if needed.
    """
    batch_sizes = _get_batch_sizes_to_try(predict_batch_size)

    # Try on default device
    result = _try_evaluate_with_fallback(
        model, X_test, y_test, vocab, batch_sizes, device="default"
    )
    if result is not None:
        return result

    # Try CPU fallback
    if not allow_cpu_fallback:
        raise RuntimeError("Evaluation failed on default device and CPU fallback disabled.")

    if model_type is None or config is None:
        raise RuntimeError("CPU fallback requested, but model_type/config were not provided.")

    print("GPU prediction failed. Falling back to CPU...")

    # Lazy import to avoid circular dependency
    from training.models import clone_model_to_cpu

    cpu_model = clone_model_to_cpu(model, model_type, config, use_gpu)

    result = _try_evaluate_with_fallback(
        cpu_model, X_test, y_test, vocab, batch_sizes, device="cpu"
    )
    if result is not None:
        return result

    raise RuntimeError("Evaluation failed on both default device and CPU.")


def _get_batch_sizes_to_try(initial_batch_size: int) -> list[int]:
    """Get ordered list of batch sizes to try."""
    batch_sizes = [initial_batch_size]
    for bs in FALLBACK_BATCH_SIZES:
        if bs not in batch_sizes:
            batch_sizes.append(bs)
    return batch_sizes


def _try_evaluate_with_fallback(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    vocab: Any,
    batch_sizes: list[int],
    device: str,
) -> dict[str, Any] | None:
    """Try evaluation with progressively smaller batch sizes."""
    device_context = tf.device("/CPU:0") if device == "cpu" else _null_context()

    for bs in batch_sizes:
        try:
            with device_context:
                print(f"Evaluating on {device} with batch_size={bs}")
                preds = model.predict(X_test, batch_size=bs, verbose=0)

            y_pred = np.argmax(preds, axis=-1)
            metrics = compute_metrics(y_test, y_pred, vocab)

            metrics["predict_batch_size_used"] = int(bs)
            metrics["predict_device"] = device if device == "cpu" else "gpu"

            return metrics

        except tf.errors.ResourceExhaustedError:
            print(f"OOM on {device} with batch_size={bs}. Trying smaller batch...")

    return None


class _null_context:
    """No-op context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
