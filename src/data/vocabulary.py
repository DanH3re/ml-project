from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Iterable, Mapping, TypeAlias

Tokens: TypeAlias = Sequence[str]
Tags: TypeAlias = Sequence[str]
Sentence: TypeAlias = tuple[Tokens, Tags]
Dataset: TypeAlias = Iterable[Sentence]

BROWN_TO_UPOS = {
    # Nouns
    "NN": "NOUN", "NNS": "NOUN", "NP": "PROPN", "NPS": "PROPN",

    # Verbs
    "VB": "VERB", "VBD": "VERB", "VBG": "VERB",
    "VBN": "VERB", "VBP": "VERB", "VBZ": "VERB",

    # Adjectives / Adverbs
    "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ",
    "RB": "ADV", "RBR": "ADV", "RBS": "ADV",

    # Determiners / Pronouns
    "AT": "DET", "DT": "DET",
    "PRP": "PRON", "PP$": "PRON",

    # Prepositions / conjunctions
    "IN": "ADP",
    "CC": "CCONJ",

    # Numbers
    "CD": "NUM",

    # Misc
    "UH": "INTJ",
    ".": "PUNCT", ",": "PUNCT", ":": "PUNCT",
}


@dataclass(frozen=True)
class Encoding:
    word_ids: list[int]
    tag_ids: list[int]


@dataclass(frozen=True)
class Vocabulary:
    word2id: Mapping[str, int]
    tag2id: Mapping[str, int]
    id2word: Sequence[str]
    id2tag: Sequence[str]

    def encode(self, tokens: Tokens, tags: Tags) -> Encoding:
        assert len(tokens) == len(tags)
        unk_id = self.word2id["<UNK>"]
        word_ids = [self.word2id.get(token, unk_id) for token in tokens]
        tag_ids = [self.tag2id[tag] for tag in tags]
        return Encoding(word_ids, tag_ids)

    def decode(self, encoding: Encoding) -> Sentence:
        tokens = [self.id2word[word_id] for word_id in encoding.word_ids]
        tags = [self.id2tag[tag_id] for tag_id in encoding.tag_ids]
        return tokens, tags


def preprocess_tags(tags: Tags) -> Tags:
    processed = []
    for tag in tags:
        base_tag = tag.split("-")[0]
        processed.append(base_tag)
    return processed


def preprocess_tokens(tokens: Tokens) -> Tokens:
    return [token.lower().strip() for token in tokens]


def build_vocab(dataset: Dataset) -> Vocabulary:
    word_set = set()
    tag_set = set()

    for tokens, tags in dataset:
        word_set.update(tokens)
        tag_set.update(tags)

    word_list = ["<PAD>", "<UNK>"] + sorted(word_set)
    tag_list = ["<PAD>"] + sorted(tag_set)

    word2id = {w: i for i, w in enumerate(word_list)}
    tag2id = {t: i for i, t in enumerate(tag_list)}

    return Vocabulary(word2id, tag2id, word_list, tag_list)
