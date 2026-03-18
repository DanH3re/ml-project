"""Metrics calculation including per-tag statistics."""
from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    vocab: Any = None,
) -> dict[str, Any]:
    """Compute overall and per-tag metrics."""
    # Flatten and mask padding tokens (tag_id = 0)
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    mask = y_true_flat != 0
    y_true_masked = y_true_flat[mask]
    y_pred_masked = y_pred_flat[mask]

    # Overall metrics
    acc = accuracy_score(y_true_masked, y_pred_masked)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_masked, y_pred_masked, average="weighted", zero_division=0
    )

    metrics = {
        "token_accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }

    # Per-tag statistics
    tag_stats = compute_per_tag_stats(y_true_masked, y_pred_masked, vocab)
    metrics["per_tag"] = tag_stats

    # Tag distribution counts
    metrics["tag_counts"] = compute_tag_counts(y_true_masked, y_pred_masked, vocab)

    return metrics


def compute_per_tag_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    vocab: Any = None,
) -> dict[str, dict[str, float]]:
    """Compute precision, recall, F1, and support for each tag."""
    unique_tags = np.unique(np.concatenate([y_true, y_pred]))

    prec_per_tag, rec_per_tag, f1_per_tag, support = precision_recall_fscore_support(
        y_true, y_pred, labels=unique_tags, zero_division=0
    )

    per_tag = {}
    for i, tag_id in enumerate(unique_tags):
        tag_name = _get_tag_name(tag_id, vocab)
        per_tag[tag_name] = {
            "tag_id": int(tag_id),
            "precision": float(prec_per_tag[i]),
            "recall": float(rec_per_tag[i]),
            "f1": float(f1_per_tag[i]),
            "support": int(support[i]),
        }

    return per_tag


def compute_tag_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    vocab: Any = None,
) -> dict[str, Any]:
    """Compute tag distribution counts for true and predicted labels."""
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)

    # Convert to tag names if vocab is available
    true_dist = {}
    pred_dist = {}
    confusion = {}

    for tag_id in set(true_counts.keys()) | set(pred_counts.keys()):
        tag_name = _get_tag_name(tag_id, vocab)
        true_dist[tag_name] = int(true_counts.get(tag_id, 0))
        pred_dist[tag_name] = int(pred_counts.get(tag_id, 0))

        # Compute difference (predicted - true)
        diff = pred_counts.get(tag_id, 0) - true_counts.get(tag_id, 0)
        if diff != 0:
            confusion[tag_name] = {
                "true_count": int(true_counts.get(tag_id, 0)),
                "pred_count": int(pred_counts.get(tag_id, 0)),
                "difference": int(diff),
                "over_predicted": diff > 0,
            }

    return {
        "true_distribution": true_dist,
        "pred_distribution": pred_dist,
        "total_true_tokens": int(len(y_true)),
        "total_pred_tokens": int(len(y_pred)),
        "unique_true_tags": len(true_counts),
        "unique_pred_tags": len(pred_counts),
        "confusion_summary": confusion,
    }


def _get_tag_name(tag_id: int, vocab: Any) -> str:
    """Get tag name from vocab or return string representation of ID."""
    if vocab is not None and hasattr(vocab, "id2tag"):
        id2tag = vocab.id2tag
        # Handle both list and dict types
        if isinstance(id2tag, list):
            if 0 <= tag_id < len(id2tag):
                return id2tag[tag_id]
        elif isinstance(id2tag, dict):
            return id2tag.get(tag_id, f"TAG_{tag_id}")
    return f"TAG_{tag_id}"


def print_tag_statistics(metrics: dict[str, Any]) -> None:
    """Print a summary of per-tag statistics."""
    tag_counts = metrics.get("tag_counts", {})
    per_tag = metrics.get("per_tag", {})

    print("\n" + "=" * 60)
    print("TAG STATISTICS")
    print("=" * 60)

    print(f"\nTotal tokens: {tag_counts.get('total_true_tokens', 'N/A')}")
    print(f"Unique true tags: {tag_counts.get('unique_true_tags', 'N/A')}")
    print(f"Unique predicted tags: {tag_counts.get('unique_pred_tags', 'N/A')}")

    # Get distributions for true/pred counts
    true_dist = tag_counts.get("true_distribution", {})
    pred_dist = tag_counts.get("pred_distribution", {})

    # Sort by true count (most common first)
    sorted_tags = sorted(
        per_tag.items(),
        key=lambda x: true_dist.get(x[0], 0),
        reverse=True,
    )

    print("\nPer-tag performance (sorted by frequency):")
    print("-" * 80)
    print(f"{'Tag':<12} {'True':>7} {'Pred':>7} {'Diff':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 80)

    for tag_name, stats in sorted_tags[:20]:  # Show top 20
        true_count = true_dist.get(tag_name, 0)
        pred_count = pred_dist.get(tag_name, 0)
        diff = pred_count - true_count
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(
            f"{tag_name:<12} "
            f"{true_count:>7} "
            f"{pred_count:>7} "
            f"{diff_str:>7} "
            f"{stats['precision']:>7.4f} "
            f"{stats['recall']:>7.4f} "
            f"{stats['f1']:>7.4f}"
        )

    # Show confusion summary (over/under predictions)
    confusion = tag_counts.get("confusion_summary", {})
    if confusion:
        print("\nPrediction bias (over/under-predicted tags):")
        print("-" * 60)
        sorted_confusion = sorted(
            confusion.items(),
            key=lambda x: abs(x[1]["difference"]),
            reverse=True,
        )

        for tag_name, info in sorted_confusion[:10]:
            direction = "over" if info["over_predicted"] else "under"
            print(
                f"  {tag_name}: {direction}-predicted by {abs(info['difference'])} "
                f"(true={info['true_count']}, pred={info['pred_count']})"
            )

    print("=" * 60)
