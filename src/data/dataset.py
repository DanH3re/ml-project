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


def load_brown() -> tuple[list[Sentence], Vocabulary, list[Encoding]]:
    nltk.download("brown", quiet=True)
    sentences = brown.tagged_sents()

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
