#!/usr/bin/env python3
"""
Count actual sentence totals for datasets supported by this project.
Uses the same loader path as training, so counts match what your pipeline sees.
"""

import argparse
import csv
import sys
from pathlib import Path

# Make imports work when this script is placed in src/tests/
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import load_dataset_by_name
from data.dataset import GLOBAL_SPLIT_SEED, GLOBAL_TEST_SIZE, _load_or_create_global_split

DEFAULT_DATASETS = ["ud", "brown"]

def count_dataset(dataset_name: str):
    """Count sentences/vocab/tags for a full dataset."""
    raw_data, vocab, encoded = load_dataset_by_name(dataset_name=dataset_name, n=None)
    global_split = _load_or_create_global_split(dataset_name, len(raw_data))

    # raw_data and encoded should have equal lengths; raw_data is clearer for sentence count
    return {
        "dataset": dataset_name,
        "sentences": len(raw_data),
        "vocab_size": len(vocab.id2word),
        "num_tags": len(vocab.id2tag),
        "global_split_seed": GLOBAL_SPLIT_SEED,
        "global_test_ratio": GLOBAL_TEST_SIZE,
        "global_train_pool_size": int(len(global_split["train_indices"])),
        "global_test_size": int(len(global_split["test_indices"])),
        "global_split_path": str(global_split["path"]),
    }

def main():
    parser = argparse.ArgumentParser(description="Count dataset sentence totals")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to inspect (default: all supported)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV output path",
    )
    args = parser.parse_args()

    rows = []
    for ds in args.datasets:
        try:
            row = count_dataset(ds)
            rows.append(row)
            print(
                f"{row['dataset']:10s} "
                f"sentences={row['sentences']:7d} "
                f"vocab={row['vocab_size']:7d} tags={row['num_tags']:3d} "
                f"split_seed={row['global_split_seed']} test_ratio={row['global_test_ratio']:.2f} "
                f"train_pool={row['global_train_pool_size']:7d} test={row['global_test_size']:7d}"
            )
            print(f"{'':10s} split_file={row['global_split_path']}")
        except Exception as e:
            print(f"{ds:10s} FAILED: {e}")

    print("\nTotals by dataset:")
    for r in sorted(rows, key=lambda x: x["dataset"]):
        print(f"{r['dataset']:10s} total_sentences={r['sentences']}")

    if args.csv:
        fieldnames = [
            "dataset",
            "sentences",
            "vocab_size",
            "num_tags",
            "global_split_seed",
            "global_test_ratio",
            "global_train_pool_size",
            "global_test_size",
            "global_split_path",
        ]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote CSV to: {args.csv}")

if __name__ == "__main__":
    main()