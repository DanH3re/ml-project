"""Training package for POS tagging models.

This package provides modular components for training and evaluating
POS tagging models with transformer and LSTM architectures.

Modules:
    config: Configuration loading and defaults
    layers: Custom Keras layers
    models: Model building functions
    trainer: Training orchestration with OOM fallback
    utils: Utility functions
"""

from .config import load_configs, print_config_summary
from .layers import LinearWarmup, TokenAndPositionEmbedding, TransformerBlock
from .models import build_model, clone_model_to_cpu
from .trainer import train_one_config
from .utils import configure_runtime, make_json_safe, set_seed, slugify
from .ranking import _DependencyResolver, _expand_pick_best_dependencies


__all__ = [
    # Config
    "load_configs",
    "print_config_summary",
    # Layers
    "LinearWarmup",
    "TokenAndPositionEmbedding",
    "TransformerBlock",
    # Models
    "build_model",
    "clone_model_to_cpu",
    # Trainer
    "train_one_config",
    # Utils
    "configure_runtime",
    "make_json_safe",
    "set_seed",
    "slugify",
    # Ranking
    "_DependencyResolver",
    "_expand_pick_best_dependencies",
]
