from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tensorflow.keras import Model
    from src.data.vocabulary import Vocabulary, Encoding
    from src.data.vocabulary import Sentence


@dataclass
class Context:
    # Config
    vocab_size: int = 1000
    d: int = 128
    max_seq_length: int = 20
    use_brown: bool = False

    # Outputs populated by steps
    preprocessed: Optional[list] = field(default=None, repr=False)
    vocab: Optional[Vocabulary] = field(default=None, repr=False)
    encoded: Optional[list[Encoding]] = field(default=None, repr=False)
    model: Optional[Model] = field(default=None, repr=False)
