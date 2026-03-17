#!/usr/bin/env python3
"""
Quick test script to verify all datasets load correctly.
This helps catch issues before running full training experiments.
"""

from data.dataset import load_dataset_by_name


def test_dataset(name: str, n: int = 10) -> None:
    """Test loading a small sample from a dataset."""
    print(f"\n{'='*60}")
    print(f"Testing dataset: {name}")
    print('='*60)

    try:
        raw_data, vocab, encoded = load_dataset_by_name(name, n=n)

        print(f"✓ Loaded {len(raw_data)} sentences")
        print(f"✓ Vocabulary size: {len(vocab.id2word)} words")
        print(f"✓ Tag set size: {len(vocab.id2tag)} tags")
        print(f"✓ Tags: {', '.join(vocab.id2tag)}")

        if raw_data:
            tokens, tags = raw_data[0]
            print(f"\nExample sentence:")
            print(f"  Tokens: {' '.join(tokens[:10])}{'...' if len(tokens) > 10 else ''}")
            print(f"  Tags:   {' '.join(tags[:10])}{'...' if len(tags) > 10 else ''}")

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    datasets = ["ud", "brown", "conll2003", "ptb", "gum"]

    print("=" * 60)
    print("POS Tagging Dataset Loader Test")
    print("=" * 60)

    results = {}
    for dataset_name in datasets:
        results[dataset_name] = test_dataset(dataset_name, n=100)

    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)

    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} {name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nPassed: {passed}/{total}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
