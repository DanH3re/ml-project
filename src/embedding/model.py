from tensorflow.keras.models import Model
from src.embedding.layers import build_embedding_layers


def build_model(vocab_size: int, d: int, max_seq_length: int) -> Model:
    inputs, x = build_embedding_layers(vocab_size, d, max_seq_length)
    return Model(inputs, x)
