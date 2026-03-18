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

DEFAULT_DATASETS = ["ud", "brown"]

# Datasets are loaded as full corpora in the loader; split is app-level.
SPLITS_BY_DATASET = {
    "ud": [None],
    "brown": [None],
}

def count_dataset(dataset_name: str, split: str | None):
    kwargs = {"dataset_name": dataset_name, "n": None}
    if split is not None:
        kwargs["split"] = split

    raw_data, vocab, encoded = load_dataset_by_name(**kwargs)

    # raw_data and encoded should have equal lengths; raw_data is clearer for sentence count
    return {
        "dataset": dataset_name,
        "split": split if split is not None else "all",
        "sentences": len(raw_data),
        "vocab_size": len(vocab.id2word),
        "num_tags": len(vocab.id2tag),
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
        splits = SPLITS_BY_DATASET.get(ds, [None])
        for split in splits:
            try:
                row = count_dataset(ds, split)
                rows.append(row)
                print(
                    f"{row['dataset']:10s} split={row['split']:10s} "
                    f"sentences={row['sentences']:7d} "
                    f"vocab={row['vocab_size']:7d} tags={row['num_tags']:3d}"
                )
            except Exception as e:
                print(f"{ds:10s} split={str(split):10s} FAILED: {e}")

    # Also print totals across splits for split-aware datasets
    print("\nTotals by dataset (sum of reported splits):")
    totals = {}
    for r in rows:
        totals.setdefault(r["dataset"], 0)
        totals[r["dataset"]] += r["sentences"]
    for ds in sorted(totals):
        print(f"{ds:10s} total_sentences={totals[ds]}")

    if args.csv:
        fieldnames = ["dataset", "split", "sentences", "vocab_size", "num_tags"]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote CSV to: {args.csv}")

if __name__ == "__main__":
    main()