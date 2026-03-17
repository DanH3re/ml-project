"""Utility functions for training pipeline."""
import random
import re
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import tensorflow as tf


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_runtime(use_gpu: str) -> None:
    """Configure TensorFlow runtime for GPU or CPU."""
    if use_gpu not in {"amd", "nvidia", "none"}:
        raise ValueError("use_gpu must be one of: 'amd', 'nvidia', 'none'")

    if use_gpu == "none":
        _disable_gpu()
    else:
        _enable_gpu()

    _print_runtime_info()


def _disable_gpu() -> None:
    """Disable GPU and use CPU only."""
    try:
        tf.config.set_visible_devices([], "GPU")
        print("GPU disabled. CPU-only mode enabled.")
    except Exception as e:
        print(f"Could not disable GPU: {e}")


def _enable_gpu() -> None:
    """Enable GPU with memory growth."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU detected.")
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(f"Could not set memory growth for {gpu}: {e}")

    print("Visible GPUs:", gpus)


def _print_runtime_info() -> None:
    """Print TensorFlow runtime information."""
    print("TensorFlow version:", tf.__version__)
    print("Physical GPUs:", tf.config.list_physical_devices("GPU"))
    print("Logical devices:", tf.config.list_logical_devices())


def slugify(name: str) -> str:
    """Convert a name to a safe filename."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")


def make_json_safe(obj: Any) -> Any:
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if is_dataclass(obj):
        return make_json_safe(asdict(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
