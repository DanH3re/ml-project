from __future__ import annotations

from typing import Any

import nltk
from nltk.corpus import brown
from datasets import load_dataset
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


def load_brown(n: int | None = None) -> tuple[list[Sentence], Vocabulary, list[Encoding]]:
    nltk.download("brown", quiet=True)
    sentences = brown.tagged_sents()

    if n is not None:
        sentences = sentences[:n]

    preprocessed = []
    for sent in sentences:
        tokens, tags = zip(*sent)
        tokens = preprocess_tokens(tokens)
        tags = preprocess_tags(tags)
        preprocessed.append((tokens, tags))

    vocab = build_vocab(preprocessed)
    encoded = [vocab.encode(tokens, tags) for tokens, tags in preprocessed]
    return preprocessed, vocab, encoded


def load_ud(split: str = "train", n: int | None = 1000) -> tuple[list[Sentence], Vocabulary, list[Encoding]]:
    dataset = load_dataset("universal_dependencies", "en_ewt")
    feature = dataset[split].features["upos"].feature
    label_names = feature.names

    split_data = dataset[split]
    if n is not None:
        n = min(n, len(split_data))
    items = split_data if n is None else split_data.select(range(n))

    preprocessed = []
    for item in items:
        tokens = preprocess_tokens(item["tokens"])
        tags = [label_names[i] for i in item["upos"]]
        preprocessed.append((tokens, tags))

    vocab = build_vocab(preprocessed)
    encoded = [vocab.encode(tokens, tags) for tokens, tags in preprocessed]
    return preprocessed, vocab, encoded


def load_conll2003(split: str = "train", n: int | None = 1000) -> tuple[list[Sentence], Vocabulary, list[Encoding]]:
    """Load CoNLL-2003 dataset (primarily for NER, but has POS tags too)."""
    dataset = load_dataset("conll2003")

    split_data = dataset[split]
    if n is not None:
        n = min(n, len(split_data))
    items = split_data if n is None else split_data.select(range(n))

    preprocessed = []
    for item in items:
        tokens = preprocess_tokens(item["tokens"])
        tags = [str(tag) for tag in item["pos_tags"]]
        preprocessed.append((tokens, tags))

    vocab = build_vocab(preprocessed)
    encoded = [vocab.encode(tokens, tags) for tokens, tags in preprocessed]
    return preprocessed, vocab, encoded


def load_ptb(n: int | None = 1000) -> tuple[list[Sentence], Vocabulary, list[Encoding]]:
    """Load Penn Treebank via NLTK (requires treebank corpus)."""
    try:
        nltk.download("treebank", quiet=True)
        from nltk.corpus import treebank

        sentences = treebank.tagged_sents()
        if n is not None:
            sentences = sentences[:n]

        preprocessed = []
        for sent in sentences:
            tokens, tags = zip(*sent)
            tokens = preprocess_tokens(tokens)
            tags = preprocess_tags(tags)
            preprocessed.append((tokens, tags))

        vocab = build_vocab(preprocessed)
        encoded = [vocab.encode(tokens, tags) for tokens, tags in preprocessed]
        return preprocessed, vocab, encoded
    except Exception as e:
        raise RuntimeError(f"Failed to load Penn Treebank: {e}")


def load_gum(split: str = "train", n: int | None = 1000) -> tuple[list[Sentence], Vocabulary, list[Encoding]]:
    """Load GUM (Georgetown University Multilayer) corpus from Universal Dependencies."""
    dataset = load_dataset("universal_dependencies", "en_gum")
    feature = dataset[split].features["upos"].feature
    label_names = feature.names

    split_data = dataset[split]
    if n is not None:
        n = min(n, len(split_data))
    items = split_data if n is None else split_data.select(range(n))

    preprocessed = []
    for item in items:
        tokens = preprocess_tokens(item["tokens"])
        tags = [label_names[i] for i in item["upos"]]
        preprocessed.append((tokens, tags))

    vocab = build_vocab(preprocessed)
    encoded = [vocab.encode(tokens, tags) for tokens, tags in preprocessed]
    return preprocessed, vocab, encoded


def load_tweets(split: str = "train", n: int | None = 1000) -> tuple[list[Sentence], Vocabulary, list[Encoding]]:
    """Load English Tweets from Universal Dependencies (Tweebank)."""
    dataset = load_dataset("universal_dependencies", "en_ewt")  # Using en_ewt as fallback
    # Note: For actual Tweebank, you would use "en_tweet" or similar
    # This is a placeholder - update with actual tweet corpus when available
    feature = dataset[split].features["upos"].feature
    label_names = feature.names

    split_data = dataset[split]
    if n is not None:
        n = min(n, len(split_data))
    items = split_data if n is None else split_data.select(range(n))

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
    n: int | None = 1000
) -> tuple[list[Sentence], Vocabulary, list[Encoding]]:
    """
    Load a dataset by name.

    Args:
        dataset_name: One of 'ud', 'brown', 'conll2003', 'ptb', 'gum', 'tweets'
        split: Dataset split (train/validation/test). Only used for datasets that have splits.
        n: Number of sentences to load. None = all sentences.

    Returns:
        (preprocessed_sentences, vocabulary, encoded_data)
    """
    loaders = {
        "ud": lambda: load_ud(split=split, n=n),
        "brown": lambda: load_brown(n=n),
        "conll2003": lambda: load_conll2003(split=split, n=n),
        "ptb": lambda: load_ptb(n=n),
        "gum": lambda: load_gum(split=split, n=n),
        "tweets": lambda: load_tweets(split=split, n=n),
    }

    if dataset_name not in loaders:
        available = ", ".join(loaders.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")

    return loaders[dataset_name]()


class DatasetCache:
    """Cache for loaded datasets to avoid redundant loading."""

    def __init__(self):
        self._cache: dict[tuple[str, int | str], dict[str, Any]] = {}

    def get(self, dataset_name: str, sentences: int | str) -> dict[str, Any]:
        """Get or load a dataset."""
        cache_key = (dataset_name, sentences)

        if cache_key not in self._cache:
            self._cache[cache_key] = self._load(dataset_name, sentences)

        return self._cache[cache_key]

    def _load(self, dataset_name: str, sentences: int | str) -> dict[str, Any]:
        """Load a dataset from source."""
        n = None if sentences == "max" else int(sentences)
        label = "all" if sentences == "max" else sentences
        print(f"\nLoading dataset={dataset_name}, sentences={label} ...")

        raw_data, vocab, encoded = load_dataset_by_name(dataset_name, n=n)

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
    sentences = config["sentences"]
    dataset_name = config.get("dataset", "ud")
    maxlen = int(config["maxlen"])
    split_seed = int(config["split_seed"])

    ds = dataset_cache.get(dataset_name, sentences)

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
