#!/usr/bin/env python3
import argparse
import json
import random
import re
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

from data.dataset import load_dataset_by_name

SAVE_MODELS = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_runtime(use_gpu: str) -> None:
    if use_gpu not in {"amd", "nvidia", "none"}:
        raise ValueError("use_gpu must be one of: 'amd', 'nvidia', 'none'")

    if use_gpu == "none":
        try:
            tf.config.set_visible_devices([], "GPU")
            print("GPU disabled. CPU-only mode enabled.")
        except Exception as e:
            print(f"Could not disable GPU: {e}")
    else:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            print(f"Requested GPU mode '{use_gpu}', but no GPU was detected.")
        else:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception as e:
                    print(f"Could not set memory growth for {gpu}: {e}")
            print("Visible GPUs:", gpus)

    print("TensorFlow version:", tf.__version__)
    print("Physical GPUs:", tf.config.list_physical_devices("GPU"))
    print("Logical devices:", tf.config.list_logical_devices())


def prepare_for_keras(encoded_list, maxlen: int) -> tuple[np.ndarray, np.ndarray]:
    word_ids = [e.word_ids for e in encoded_list]
    tag_ids = [e.tag_ids for e in encoded_list]
    X = pad_sequences(word_ids, maxlen=maxlen, padding="post", value=0)
    y = pad_sequences(tag_ids, maxlen=maxlen, padding="post", value=0)
    return np.array(X), np.array(y)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int):
        super().__init__()
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
        config.update({"maxlen": self.maxlen, "vocab_size": self.vocab_size, "embed_dim": self.embed_dim})
        return config


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1):
        super().__init__()
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
        # Keras auto-propagates query_mask/value_mask to MHA from the
        # embedding's _keras_mask, so no explicit attention_mask needed.
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def compute_mask(self, inputs, mask=None):
        return mask  # pass-through so every stacked block receives the padding mask

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate})
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


def build_model(model_type: str, config: dict[str, Any], use_gpu: str) -> keras.Model:
    inputs = layers.Input(shape=(config["maxlen"],), dtype="int32")

    if model_type == "transformer":
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

    elif model_type == "lstm":
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
    else:
        raise ValueError("model_type must be 'transformer' or 'lstm'")

    outputs = layers.Dense(config["num_tags"], activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    warmup_steps = config.get("lr_warmup_steps", 0)
    lr = LinearWarmup(config["lr"], warmup_steps) if warmup_steps > 0 else config["lr"]
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def clone_model_on_cpu(
    trained_model: keras.Model,
    model_type: str,
    config: dict[str, Any],
    use_gpu: str,
) -> keras.Model:
    with tf.device("/CPU:0"):
        cpu_model = build_model(model_type, config, use_gpu=use_gpu)
        cpu_model.set_weights(trained_model.get_weights())
    return cpu_model

def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    predict_batch_size: int = 32,
    allow_cpu_fallback: bool = True,
    model_type: str | None = None,
    config: dict[str, Any] | None = None,
    use_gpu: str = "none",
) -> dict[str, float]:
    batch_sizes_to_try = []
    for bs in [predict_batch_size, 16, 8, 4, 2, 1]:
        if bs not in batch_sizes_to_try:
            batch_sizes_to_try.append(bs)

    last_error = None

    # First try GPU / default model
    for bs in batch_sizes_to_try:
        try:
            print(f"Evaluating with default device, batch_size={bs}")
            preds = model.predict(X_test, batch_size=bs, verbose=0)

            y_pred = np.argmax(preds, axis=-1).flatten()
            y_true = y_test.flatten()

            mask = y_true != 0
            y_true = y_true[mask]
            y_pred = y_pred[mask]

            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0
            )
            return {
                "token_accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "predict_batch_size_used": int(bs),
                "predict_device": "gpu",
            }
        except tf.errors.ResourceExhaustedError as e:
            print(f"Prediction failed on default device with batch_size={bs}. Retrying smaller batch.")
            last_error = e

    if not allow_cpu_fallback:
        raise last_error

    if model_type is None or config is None:
        raise RuntimeError("CPU fallback requested, but model_type/config were not provided.")

    print("GPU prediction failed at all tried batch sizes. Rebuilding model on CPU for fallback.")
    cpu_model = clone_model_on_cpu(model, model_type=model_type, config=config, use_gpu=use_gpu)

    for bs in batch_sizes_to_try:
        try:
            with tf.device("/CPU:0"):
                print(f"Evaluating on CPU with batch_size={bs}")
                preds = cpu_model.predict(X_test, batch_size=bs, verbose=0)

            y_pred = np.argmax(preds, axis=-1).flatten()
            y_true = y_test.flatten()

            mask = y_true != 0
            y_true = y_true[mask]
            y_pred = y_pred[mask]

            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0
            )
            return {
                "token_accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "predict_batch_size_used": int(bs),
                "predict_device": "cpu",
            }
        except Exception as e:
            print(f"CPU prediction also failed with batch_size={bs}: {e}")
            last_error = e

    raise last_error


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if is_dataclass(obj):
        return make_json_safe(asdict(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def default_configs(max_len: int, sentences: int, seed: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "transformer_small",
            "model_type": "transformer",
            "sentences": sentences,
            "maxlen": max_len,
            "split_seed": seed,
            "seed": seed,
            "embed_dim": 128,
            "num_heads": 4,
            "ff_dim": 128,
            "num_layers": 1,
            "dropout": 0.1,
            "lr": 0.001,
            "epochs": 5,
            "batch_size": 32,
        },
        {
            "name": "lstm_baseline",
            "model_type": "lstm",
            "sentences": sentences,
            "maxlen": max_len,
            "split_seed": seed,
            "seed": seed,
            "embed_dim": 128,
            "lstm_units": 64,
            "lr": 0.001,
            "epochs": 10,
            "batch_size": 32,
        },
    ]


def load_configs(config_path: str | None, max_len: int, sentences: int, seed: int) -> list[dict[str, Any]]:
    if config_path is None:
        return default_configs(max_len, sentences, seed)

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        configs = [data]
    elif isinstance(data, list):
        configs = data
    else:
        raise ValueError("Config JSON must be either an object or a list of objects.")

    for cfg in configs:
        cfg.setdefault("sentences", sentences)
        cfg.setdefault("maxlen", max_len)
        cfg.setdefault("split_seed", seed)
        cfg.setdefault("seed", seed)

    return configs


def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")


def get_dataset_for_sentences(
    sentences: int | str,
    dataset_name: str,
    dataset_cache: dict[tuple[str, int | str], dict[str, Any]],
) -> dict[str, Any]:
    cache_key = (dataset_name, sentences)
    if cache_key not in dataset_cache:
        n = None if sentences == "max" else int(sentences)
        label = "all" if sentences == "max" else sentences
        print(f"\nLoading dataset={dataset_name}, sentences={label} ...")
        raw_data, vocab, encoded = load_dataset_by_name(dataset_name, n=n)
        dataset_cache[cache_key] = {
            "raw_data": raw_data,
            "vocab": vocab,
            "encoded": encoded,
            "actual_sentences": len(encoded),
        }
    return dataset_cache[cache_key]


def prepare_split_for_config(
    config: dict[str, Any],
    dataset_cache: dict[tuple[str, int | str], dict[str, Any]],
) -> dict[str, Any]:
    sentences = config["sentences"]  # int or "max"
    dataset_name = config.get("dataset", "ud")  # default to UD for backwards compatibility
    maxlen = int(config["maxlen"])
    split_seed = int(config["split_seed"])

    ds = get_dataset_for_sentences(sentences, dataset_name, dataset_cache)
    raw_data = ds["raw_data"]
    vocab = ds["vocab"]
    encoded = ds["encoded"]

    X, y = prepare_for_keras(encoded, maxlen)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=split_seed
    )

    prepared = {
        "raw_data": raw_data,
        "vocab": vocab,
        "encoded": encoded,
        "actual_sentences": ds["actual_sentences"],
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "vocab_size": len(vocab.id2word),
        "num_tags": len(vocab.id2tag),
        "dataset_shape": {
            "X": tuple(X.shape),
            "y": tuple(y.shape),
            "X_train": tuple(X_train.shape),
            "X_test": tuple(X_test.shape),
            "y_train": tuple(y_train.shape),
            "y_test": tuple(y_test.shape),
        },
    }
    return prepared


def train_one_config(
    config: dict[str, Any],
    use_gpu: str,
    models_dir: Path,
    dataset_cache: dict[tuple[str, int | str], dict[str, Any]],
) -> dict[str, Any]:
    name = config.get("name", f"{config['model_type']}_run")
    run_seed = int(config.get("seed", config.get("split_seed", 42)))
    set_seed(run_seed)

    prepared = prepare_split_for_config(config, dataset_cache)

    config = dict(config)
    config["vocab_size"] = prepared["vocab_size"]
    config["num_tags"] = prepared["num_tags"]

    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]

    print("\n" + "=" * 80)
    print(f"Training run: {name}")
    print(json.dumps(make_json_safe(config), indent=2))
    print("Dataset shapes:")
    print(json.dumps(make_json_safe(prepared["dataset_shape"]), indent=2))
    print("=" * 80)

    keras.backend.clear_session()
    model = build_model(config["model_type"], config, use_gpu)
    model.summary()

    callbacks: list[keras.callbacks.Callback] = []
    if config.get("early_stopping", True):
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.get("patience", 3),
                restore_best_weights=True,
            )
        )

    start = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        batch_size=config.get("batch_size", 32),
        epochs=config["epochs"],
        validation_split=config.get("validation_split", 0.2),
        verbose=1,
        callbacks=callbacks,
    )
    train_time_sec = time.perf_counter() - start

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        predict_batch_size=int(config.get("predict_batch_size", 32)),
        model_type=config["model_type"],
        config=config,
        use_gpu=use_gpu,
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{slugify(name)}.keras"

    if SAVE_MODELS:
        model.save(model_path)

    result = {
        "name": name,
        "model_type": config["model_type"],
        "config": make_json_safe(config),
        "dataset_meta": {
            "dataset": config.get("dataset", "ud"),
            "sentences": int(prepared["actual_sentences"]),
            "sentences_config": config["sentences"],
            "maxlen": int(config["maxlen"]),
            "split_seed": int(config["split_seed"]),
            "seed": int(run_seed),
            "vocab_size": int(prepared["vocab_size"]),
            "num_tags": int(prepared["num_tags"]),
            "shapes": make_json_safe(prepared["dataset_shape"]),
        },
        "num_params": int(model.count_params()),
        "train_time_sec": float(train_time_sec),
        "best_val_accuracy": float(max(history.history.get("val_accuracy", [0.0]))),
        "best_val_loss": float(min(history.history.get("val_loss", [float("inf")]))),
        "epochs_ran": int(len(history.history.get("loss", []))),
        "history": make_json_safe(history.history),
        "test_metrics": metrics,
        "model_path": str(model_path),
        "model_saved": bool(SAVE_MODELS),
    }

    print("\nTest metrics:")
    print(json.dumps(result["test_metrics"], indent=2))
    if SAVE_MODELS:
        print(f"Saved model to: {model_path}")
    else:
        print(f"Model saving disabled. Would save to: {model_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train POS tagging models.")
    parser.add_argument("--use-gpu", choices=["amd", "nvidia", "none"], default="none")
    parser.add_argument("--sentences", type=int, default=2000)
    parser.add_argument("--max-len", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config or config list",
    )
    parser.add_argument("--output", type=str, default="resources/results/training_results.json")
    parser.add_argument("--models-dir", type=str, default="resources/models")
    parser.add_argument("--save-models", action="store_true", help="Actually save .keras model files")
    args = parser.parse_args()

    global SAVE_MODELS
    SAVE_MODELS = args.save_models

    set_seed(args.seed)
    configure_runtime(args.use_gpu)

    configs = load_configs(args.config, args.max_len, args.sentences, args.seed)

    print(f"\nLoaded {len(configs)} config(s).")
    unique_sentences = sorted(
        {cfg["sentences"] for cfg in configs},
        key=lambda x: float("inf") if x == "max" else x,
    )
    unique_maxlens = sorted({int(cfg["maxlen"]) for cfg in configs})
    unique_split_seeds = sorted({int(cfg["split_seed"]) for cfg in configs})
    unique_run_seeds = sorted({int(cfg["seed"]) for cfg in configs})

    print(f"Sentence-count values in sweep: {unique_sentences}")
    print(f"Max-length values in sweep: {unique_maxlens}")
    print(f"Train/test split seeds in sweep: {unique_split_seeds}")
    print(f"Run seeds in sweep: {unique_run_seeds}")

    results = []
    models_dir = Path(args.models_dir)
    dataset_cache: dict[tuple[str, int | str], dict[str, Any]] = {}

    for cfg in configs:
        result = train_one_config(
            cfg,
            args.use_gpu,
            models_dir,
            dataset_cache,
        )
        results.append(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(results), f, indent=2)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()