"""Custom Keras layers for POS tagging models."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TokenAndPositionEmbedding(layers.Layer):
    """Combined token and positional embedding layer."""

    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True,
        )
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.supports_masking = True

    def call(self, x):
        seq_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config


class TransformerBlock(layers.Layer):
    """Single transformer encoder block with multi-head attention."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.supports_masking = True

    def call(self, inputs, training=False, mask=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config


class LinearWarmup(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup to peak_lr over warmup_steps, then constant."""

    def __init__(self, peak_lr: float, warmup_steps: int):
        super().__init__()
        self.peak_lr = float(peak_lr)
        self.warmup_steps = int(warmup_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        return tf.minimum(self.peak_lr, self.peak_lr * step / tf.maximum(warmup, 1.0))

    def get_config(self):
        return {"peak_lr": self.peak_lr, "warmup_steps": self.warmup_steps}
