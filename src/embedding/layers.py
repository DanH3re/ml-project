import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Input


def build_embedding_layers(vocab_size: int, d: int, max_seq_length: int):
    inputs = Input(shape=(max_seq_length,), dtype="int32")

    token_emb_layer = Embedding(input_dim=vocab_size, output_dim=d, name="token_embedding")
    token_emb = token_emb_layer(inputs)

    scaled_token_emb = token_emb * tf.math.sqrt(tf.cast(d, tf.float32))

    seq_len = keras.ops.shape(inputs)[1]
    positions = keras.ops.arange(0, seq_len)
    positions = keras.ops.expand_dims(positions, 0)

    pos_emb_layer = Embedding(input_dim=max_seq_length, output_dim=d, name="positional_embedding")
    pos_emb = pos_emb_layer(positions)

    x = scaled_token_emb + pos_emb
    return inputs, x
