from __future__ import annotations

import nltk
from nltk.corpus import brown
from datasets import load_dataset

from data.vocabulary import (
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
