from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nltk
from nltk.corpus import brown
from datasets import load_dataset, concatenate_datasets
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .vocabulary import (
    preprocess_tokens,
    preprocess_tags,
    build_vocab,
    Sentence,
    Vocabulary,
    Encoding,
)


# Global fixed split policy (shared by all runs/configs).
GLOBAL_SPLIT_SEED = 42
GLOBAL_TEST_SIZE = 0.2
def _select_indices(
    total: int,
    n: int,
    sampling: str,
    rng: np.random.Generator,
    dataset_name: str,
    split: str | None = None,
) -> np.ndarray:
    """Select sentence indices.

    Behavior:
      - If n <= total and sampling='head': return first n indices.
      - If n <= total and sampling='random': sample n unique indices without replacement.
      - If n > total: raise ValueError.
    """
    if sampling not in {"head", "random"}:
        raise ValueError("sampling must be one of: 'head', 'random'")

    if n <= 0:
        raise ValueError("n must be a positive integer")

    if n > total:
        split_label = f" split='{split}'" if split is not None else ""
        raise ValueError(
            f"Requested n={n} for {dataset_name}{split_label}, "
            f"but maximum available is {total}. "
            "Resampling/overflow is disabled. Reduce n or switch dataset."
        )

    if sampling == "head":
        return np.arange(n)

    return rng.choice(total, size=n, replace=False)


def load_brown(
    n: int | None = None,
    sampling: str = "head",
    sampling_seed: int | None = None,
) -> tuple[list[Sentence], Vocabulary, list[Encoding]]:
    nltk.download("brown", quiet=True)
    sentences = brown.tagged_sents()

    if n is not None:
        n = int(n)
        rng = np.random.default_rng(sampling_seed)
        idx = _select_indices(
            total=len(sentences),
            n=n,
            sampling=sampling,
            rng=rng,
            dataset_name="brown",
        )
        sentences = [sentences[int(i)] for i in idx]

    preprocessed = []
    for sent in sentences:
        tokens, tags = zip(*sent)
        tokens = preprocess_tokens(tokens)
        tags = preprocess_tags(tags)
        preprocessed.append((tokens, tags))

    vocab = build_vocab(preprocessed)
    encoded = [vocab.encode(tokens, tags) for tokens, tags in preprocessed]
    return preprocessed, vocab, encoded


def load_ud(
    n: int | None = 1000,
    sampling: str = "head",
    sampling_seed: int | None = None,
) -> tuple[list[Sentence], Vocabulary, list[Encoding]]:
    dataset = load_dataset("universal_dependencies", "en_ewt")
    feature = dataset["train"].features["upos"].feature
    label_names = feature.names

    split_data = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
    if n is not None:
        n = int(n)
        rng = np.random.default_rng(sampling_seed)
        idx = _select_indices(
            total=len(split_data),
            n=n,
            sampling=sampling,
            rng=rng,
            dataset_name="ud",
            split="all",
        )
        items = split_data.select(idx.tolist())
    else:
        items = split_data

    preprocessed = []
    for item in items:
        tokens = preprocess_tokens(item["tokens"])
        tags = [label_names[i] for i in item["upos"]]
        preprocessed.append((tokens, tags))

    vocab = build_vocab(preprocessed)
    encoded = [vocab.encode(tokens, tags) for tokens, tags in preprocessed]
    return preprocessed, vocab, encoded


def load_dataset_by_name(
    dataset_name: str,
    n: int | None = 1000,
    sampling: str = "head",
    sampling_seed: int | None = None,
) -> tuple[list[Sentence], Vocabulary, list[Encoding]]:
    """
    Load a dataset by name.

    Args:
        dataset_name: One of 'ud', 'brown'
        n: Number of sentences to load. None = all sentences.
        sampling: One of 'head' or 'random'.
        sampling_seed: RNG seed used for random sampling/resampling.

    Returns:
        (preprocessed_sentences, vocabulary, encoded_data)
    """
    loaders = {
        "ud": lambda: load_ud(n=n, sampling=sampling, sampling_seed=sampling_seed),
        "brown": lambda: load_brown(n=n, sampling=sampling, sampling_seed=sampling_seed),
    }

    if dataset_name not in loaders:
        available = ", ".join(loaders.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")

    return loaders[dataset_name]()


def _project_root() -> Path:
    """Resolve repository root from this file location."""
    return Path(__file__).resolve().parents[2]


def _splits_dir() -> Path:
    """Directory where global split artifacts are stored."""
    path = _project_root() / "resources" / "datasets" / "splits"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _split_path(dataset_name: str) -> Path:
    """Build the JSON path for a persisted global split."""
    test_lbl = str(GLOBAL_TEST_SIZE).replace(".", "p")
    return _splits_dir() / f"{dataset_name}_split_seed{GLOBAL_SPLIT_SEED}_test{test_lbl}.json"


def _load_or_create_global_split(
    dataset_name: str,
    total: int,
) -> dict[str, Any]:
    """Load a persisted global split or create one if missing.

    The split is defined over full-dataset indices and reused by every config.
    """
    if not (0.0 < GLOBAL_TEST_SIZE < 1.0):
        raise ValueError("GLOBAL_TEST_SIZE must be a float in (0, 1)")
    if total < 2:
        raise ValueError("Dataset must contain at least 2 sentences to create a split")

    path = _split_path(dataset_name)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if int(payload.get("total_sentences", -1)) != int(total):
            raise ValueError(
                f"Global split file '{path}' total_sentences={payload.get('total_sentences')} "
                f"does not match current dataset size {total}."
            )

        train_indices = np.array(payload["train_indices"], dtype=np.int64)
        test_indices = np.array(payload["test_indices"], dtype=np.int64)
        return {
            "path": str(path),
            "train_indices": train_indices,
            "test_indices": test_indices,
        }

    rng = np.random.default_rng(GLOBAL_SPLIT_SEED)
    indices = np.arange(total, dtype=np.int64)
    rng.shuffle(indices)

    n_test = int(round(total * GLOBAL_TEST_SIZE))
    n_test = max(1, min(total - 1, n_test))

    test_indices = np.sort(indices[:n_test])
    train_indices = np.sort(indices[n_test:])

    payload = {
        "dataset": dataset_name,
        "split_seed": int(GLOBAL_SPLIT_SEED),
        "test_size": float(GLOBAL_TEST_SIZE),
        "total_sentences": int(total),
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Created global split file: {path}")
    return {
        "path": str(path),
        "train_indices": train_indices,
        "test_indices": test_indices,
    }


class DatasetCache:
    """Cache for loaded datasets to avoid redundant loading."""

    def __init__(self):
        self._cache: dict[str, dict[str, Any]] = {}

    def get_full(
        self,
        dataset_name: str,
    ) -> dict[str, Any]:
        """Get or load the full dataset (n=None) for a dataset name."""
        if dataset_name not in self._cache:
            self._cache[dataset_name] = self._load_full(dataset_name)
        return self._cache[dataset_name]

    def _load_full(self, dataset_name: str) -> dict[str, Any]:
        """Load full dataset from source."""
        print(f"\nLoading full dataset={dataset_name} ...")

        raw_data, vocab, encoded = load_dataset_by_name(
            dataset_name,
            n=None,
            sampling="head",
            sampling_seed=None,
        )

        return {
            "raw_data": raw_data,
            "vocab": vocab,
            "encoded": encoded,
            "actual_sentences": len(encoded),
        }


def prepare_for_keras(encoded_list, maxlen: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert encoded sentences to padded numpy arrays for Keras."""
    word_ids = [e.word_ids for e in encoded_list]
    tag_ids = [e.tag_ids for e in encoded_list]
    X = pad_sequences(word_ids, maxlen=maxlen, padding="post", value=0)
    y = pad_sequences(tag_ids, maxlen=maxlen, padding="post", value=0)
    return np.array(X), np.array(y)


def prepare_split_for_config(
    config: dict[str, Any],
    dataset_cache: DatasetCache,
) -> dict[str, Any]:
    """Prepare train/test split for a given configuration using a global frozen test set."""
    raw_sentences = config["sentences"]
    if isinstance(raw_sentences, bool) or not isinstance(raw_sentences, int):
        raise ValueError(
            f"Config '{config.get('name', '<unnamed>')}' has invalid 'sentences' value. "
            "Only positive integers are supported."
        )
    if raw_sentences <= 0:
        raise ValueError(
            f"Config '{config.get('name', '<unnamed>')}' has invalid 'sentences' value. "
            "'sentences' must be > 0."
        )

    sentences = int(raw_sentences)
    dataset_name = config.get("dataset", "ud")
    maxlen = int(config["maxlen"])
    sampling = str(config.get("sentence_sampling", "head")).strip().lower()
    sampling_seed = int(config.get("sentence_sampling_seed", GLOBAL_SPLIT_SEED))

    if sampling not in {"head", "random"}:
        raise ValueError(
            f"Config '{config.get('name', '<unnamed>')}' has invalid sentence_sampling='{sampling}'. "
            "Supported values: 'head', 'random'."
        )

    ds = dataset_cache.get_full(dataset_name)

    global_split = _load_or_create_global_split(
        dataset_name=dataset_name,
        total=len(ds["encoded"]),
    )

    train_pool_idx = global_split["train_indices"]
    test_idx = global_split["test_indices"]

    rng = np.random.default_rng(sampling_seed)
    chosen_train_local_idx = _select_indices(
        total=len(train_pool_idx),
        n=sentences,
        sampling=sampling,
        rng=rng,
        dataset_name=dataset_name,
        split="train_pool",
    )
    chosen_train_idx = train_pool_idx[chosen_train_local_idx]

    selected_idx = np.concatenate([chosen_train_idx, test_idx])
    selected_idx = selected_idx.astype(np.int64)
    selected_raw = [ds["raw_data"][int(i)] for i in selected_idx]

    # Keep existing behavior of building vocab from the selected run corpus.
    vocab = build_vocab(selected_raw)
    encoded = [vocab.encode(tokens, tags) for tokens, tags in selected_raw]
    X, y = prepare_for_keras(encoded, maxlen)

    n_train = len(chosen_train_idx)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    actual_sentences = int(len(selected_idx))

    return {
        "raw_data": selected_raw,
        "vocab": vocab,
        "encoded": encoded,
        "actual_sentences": actual_sentences,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "vocab_size": len(vocab.id2word),
        "num_tags": len(vocab.id2tag),
        "global_split_path": global_split["path"],
        "global_split_seed": int(GLOBAL_SPLIT_SEED),
        "global_test_size": float(GLOBAL_TEST_SIZE),
        "global_split_train_pool_size": int(len(train_pool_idx)),
        "global_split_test_size": int(len(test_idx)),
        "dataset_shape": {
            "X": tuple(X.shape),
            "y": tuple(y.shape),
            "X_train": tuple(X_train.shape),
            "X_test": tuple(X_test.shape),
            "y_train": tuple(y_train.shape),
            "y_test": tuple(y_test.shape),
        },
    }
