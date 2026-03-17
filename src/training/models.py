"""Model building functions for POS tagging."""
from typing import Any

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .layers import TokenAndPositionEmbedding, TransformerBlock, LinearWarmup


def build_model(model_type: str, config: dict[str, Any], use_gpu: str) -> keras.Model:
    """Build a POS tagging model based on configuration."""
    if model_type == "transformer":
        return _build_transformer(config)
    elif model_type == "lstm":
        return _build_lstm(config, use_gpu)
    else:
        raise ValueError(f"model_type must be 'transformer' or 'lstm', got: {model_type}")


def _build_transformer(config: dict[str, Any]) -> keras.Model:
    """Build a transformer-based POS tagging model."""
    inputs = layers.Input(shape=(config["maxlen"],), dtype="int32")

    x = TokenAndPositionEmbedding(
        config["maxlen"],
        config["vocab_size"],
        config["embed_dim"],
    )(inputs)

    for _ in range(config.get("num_layers", 1)):
        x = TransformerBlock(
            config["embed_dim"],
            config["num_heads"],
            config["ff_dim"],
            config.get("dropout", 0.1),
        )(x)

    outputs = layers.Dense(config["num_tags"], activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    _compile_model(model, config)
    return model


def _build_lstm(config: dict[str, Any], use_gpu: str) -> keras.Model:
    """Build an LSTM-based POS tagging model."""
    inputs = layers.Input(shape=(config["maxlen"],), dtype="int32")

    x = layers.Embedding(
        input_dim=config["vocab_size"],
        output_dim=config["embed_dim"],
        mask_zero=True,
    )(inputs)

    dropout = config.get("dropout", 0.0)
    if dropout > 0.0:
        x = layers.Dropout(dropout)(x)

    lstm_kwargs = {
        "units": config.get("lstm_units", 64),
        "return_sequences": True,
    }

    if use_gpu == "amd":
        lstm_kwargs["use_cudnn"] = False
        print("Using ROCm-safe LSTM configuration.")

    for _ in range(config.get("lstm_layers", 1)):
        x = layers.Bidirectional(layers.LSTM(**lstm_kwargs))(x)

    outputs = layers.Dense(config["num_tags"], activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    _compile_model(model, config)
    return model


def _compile_model(model: keras.Model, config: dict[str, Any]) -> None:
    """Compile model with optimizer and loss."""
    warmup_steps = config.get("lr_warmup_steps", 0)
    lr = LinearWarmup(config["lr"], warmup_steps) if warmup_steps > 0 else config["lr"]

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def clone_model_to_cpu(
    trained_model: keras.Model,
    model_type: str,
    config: dict[str, Any],
    use_gpu: str,
) -> keras.Model:
    """Clone a trained model to CPU for fallback inference."""
    with tf.device("/CPU:0"):
        cpu_model = build_model(model_type, config, use_gpu=use_gpu)
        cpu_model.set_weights(trained_model.get_weights())
    return cpu_model
