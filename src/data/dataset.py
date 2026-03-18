from __future__ import annotations

from typing import Any

import nltk
from nltk.corpus import brown
from datasets import load_dataset, concatenate_datasets
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .vocabulary import (
    preprocess_tokens,
    preprocess_tags,
    build_vocab,
    Sentence,
    Vocabulary,
    Encoding,
)
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
      - If n <= total: return first n indices (deterministic order).
      - If n > total: resample with replacement and warn.
    """
    if sampling not in {"head", "random"}:
        raise ValueError("sampling must be one of: 'head', 'random'")

    if n <= total:
        raise ValueError("_select_indices is only intended for overflow resampling")

    split_label = f" split='{split}'" if split is not None else ""
    print(
        f"WARNING: requested n={n} for {dataset_name}{split_label}, "
        f"but maximum available is {total}. "
        "Applying random resampling with replacement."
    )
    return rng.choice(total, size=n, replace=True)


def load_brown(
    n: int | None = None,
    sampling: str = "head",
    sampling_seed: int | None = None,
) -> tuple[list[Sentence], Vocabulary, list[Encoding]]:
    nltk.download("brown", quiet=True)
    sentences = brown.tagged_sents()

    if n is not None:
        n = int(n)
        if n <= len(sentences):
            sentences = sentences[:n]
        else:
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
    split: str = "train",
    n: int | None = 1000,
    sampling: str = "head",
    sampling_seed: int | None = None,
) -> tuple[list[Sentence], Vocabulary, list[Encoding]]:
    dataset = load_dataset("universal_dependencies", "en_ewt")
    feature = dataset["train"].features["upos"].feature
    label_names = feature.names

    if split != "train":
        # Kept for backward compatibility; split selection is now app-level only.
        print(
            f"WARNING: split='{split}' was provided for UD, but loader now uses the full dataset. "
            "Ignoring split and loading train+validation+test."
        )

    split_data = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
    if n is not None:
        n = int(n)
        if n <= len(split_data):
            items = split_data.select(range(n))
        else:
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
    split: str = "train",
    n: int | None = 1000,
    sampling: str = "head",
    sampling_seed: int | None = None,
) -> tuple[list[Sentence], Vocabulary, list[Encoding]]:
    """
    Load a dataset by name.

    Args:
        dataset_name: One of 'ud', 'brown'
        split: Backward-compatible parameter. UD now ignores this and always loads full data.
        n: Number of sentences to load. None = all sentences.
        sampling: One of 'head' or 'random'.
        sampling_seed: RNG seed used for random sampling/resampling.

    Returns:
        (preprocessed_sentences, vocabulary, encoded_data)
    """
    loaders = {
        "ud": lambda: load_ud(split=split, n=n, sampling=sampling, sampling_seed=sampling_seed),
        "brown": lambda: load_brown(n=n, sampling=sampling, sampling_seed=sampling_seed),
    }

    if dataset_name not in loaders:
        available = ", ".join(loaders.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")

    return loaders[dataset_name]()


class DatasetCache:
    """Cache for loaded datasets to avoid redundant loading."""

    def __init__(self):
        self._cache: dict[tuple[str, int, str, int | None], dict[str, Any]] = {}

    def get(
        self,
        dataset_name: str,
        sentences: int,
        sampling: str = "head",
        sampling_seed: int | None = None,
    ) -> dict[str, Any]:
        """Get or load a dataset."""
        cache_key = (dataset_name, sentences, sampling, sampling_seed)

        if cache_key not in self._cache:
            self._cache[cache_key] = self._load(dataset_name, sentences, sampling, sampling_seed)

        return self._cache[cache_key]

    def _load(
        self,
        dataset_name: str,
        sentences: int,
        sampling: str,
        sampling_seed: int | None,
    ) -> dict[str, Any]:
        """Load a dataset from source."""
        n = int(sentences)
        print(
            f"\nLoading dataset={dataset_name}, sentences={sentences}, "
            f"sampling={sampling} ..."
        )

        raw_data, vocab, encoded = load_dataset_by_name(
            dataset_name,
            n=n,
            sampling=sampling,
            sampling_seed=sampling_seed,
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
    """Prepare train/test split for a given configuration."""
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
    split_seed = int(config["split_seed"])
    sampling = str(config.get("sentence_sampling", "head")).strip().lower()
    sampling_seed = int(config.get("sentence_sampling_seed", config.get("seed", split_seed)))

    if sampling not in {"head", "random"}:
        raise ValueError(
            f"Config '{config.get('name', '<unnamed>')}' has invalid sentence_sampling='{sampling}'. "
            "Supported values: 'head', 'random'."
        )

    ds = dataset_cache.get(
        dataset_name,
        sentences,
        sampling=sampling,
        sampling_seed=sampling_seed,
    )

    X, y = prepare_for_keras(ds["encoded"], maxlen)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=split_seed
    )

    return {
        "raw_data": ds["raw_data"],
        "vocab": ds["vocab"],
        "encoded": ds["encoded"],
        "actual_sentences": ds["actual_sentences"],
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "vocab_size": len(ds["vocab"].id2word),
        "num_tags": len(ds["vocab"].id2tag),
        "dataset_shape": {
            "X": tuple(X.shape),
            "y": tuple(y.shape),
            "X_train": tuple(X_train.shape),
            "X_test": tuple(X_test.shape),
            "y_train": tuple(y_train.shape),
            "y_test": tuple(y_test.shape),
        },
    }
