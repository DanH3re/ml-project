"""Evaluation package for model evaluation and metrics calculation.

Modules:
    evaluation: Model evaluation with OOM fallback
    metrics: Metrics calculation with per-tag statistics
"""

from .evaluation import evaluate_model
from .metrics import (
    compute_metrics,
    compute_per_tag_stats,
    compute_tag_counts,
    print_tag_statistics,
)

__all__ = [
    "evaluate_model",
    "compute_metrics",
    "compute_per_tag_stats",
    "compute_tag_counts",
    "print_tag_statistics",
]
