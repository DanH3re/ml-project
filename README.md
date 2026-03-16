# Transformer-based POS Tagging

## Overview

This repository implements a **Transformer-based Part-of-Speech (POS) tagging system** and a **BiLSTM baseline** for comparison.  
It includes end-to-end utilities: data loading & preprocessing, vocabulary building, model definitions (Transformer and BiLSTM), training loops, evaluation metrics, and visualization tools (attention maps, confusion matrices, per-tag analysis).

Key features:
- Data preprocessing pipeline for POS-tagged corpora
- Transformer model with token + positional embeddings and multi-head self-attention
- BiLSTM baseline
- Training with configurable hyperparameters
- Token-level, sentence-level, and per-tag evaluation
- Attention visualization and error analysis

---

## System Architecture

The POS tagging pipeline:

```

Text Dataset
↓
Tokenization
↓
Vocabulary Encoding
↓
Padding & Masking
↓
Embedding Layer
↓
Positional Encoding
↓
Transformer Block / BiLSTM
↓
Classification Head
↓
POS Tag Predictions

````

The Transformer outputs a probability distribution over tags for each token. The BiLSTM returns sequence-wise tag probabilities using recurrent context.

---

## Data Processing

Before training, text is preprocessed:

- **Dataset loading:** Sentences and POS tags are extracted (examples included for Brown and Universal Dependencies).
- **Token normalization:** Lowercasing and simple stripping.
- **Tag normalization:** Splitting complex tags (e.g., `NNS-PL`) to the base tag (e.g., `NNS`).
- **Vocabulary building:** Builds `word2id`, `tag2id`, `id2word`, `id2tag`. Special tokens: `<PAD>`, `<UNK>` for words and `<PAD>` for tags.
- **Encoding:** Sentences are converted to lists of word IDs and tag IDs.
- **Padding & Masking:** Sequences are padded to `maxlen` with `0` (the `<PAD>` id) and masks are used to ignore padded positions during metrics and visualizations.

Example:

```python
sentence = ["The", "cat", "sits", "on", "the", "mat"]
tags = ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN"]

token_ids = [12, 45, 9, 3, 12, 45]
padded_ids = [12, 45, 9, 3, 12, 45, 0, 0]
mask = [1, 1, 1, 1, 1, 1, 0, 0]
````

---

## Model Components

### Token & Positional Embedding

* Token embedding: converts token IDs into dense vectors.
* Positional embedding: learned position vectors added to token embeddings so the model can reason about order.

### Transformer Block

* Multi-head self-attention that lets each position attend to every other position.
* Feed-forward network (FFN): two dense layers with non-linearity in-between.
* Residual connections + Layer Normalization around attention and FFN.
* Dropout layers for regularization.

### BiLSTM Baseline

* Embedding layer (masking enabled).
* Bidirectional LSTM with `return_sequences=True`.
* Dense softmax classification head.

### Classification Head

* Dense layer with `softmax` activation produces per-token tag probabilities.

The models are compiled with an optimizer, `sparse_categorical_crossentropy` loss, and `accuracy` metrics.

---

## Hyperparameters — what they are and why they matter

Below are the hyperparameters used in this project and descriptions, typical value ranges, effects, and tuning tips.

### Architecture & Data Hyperparameters

* `maxlen` (int)

  * **What it does:** Maximum sequence length (number of tokens) the model will process. Inputs longer than `maxlen` are truncated, shorter sequences are padded.
  * **Default in notebook:** `30`
  * **Typical range:** 16–512 (depends on dataset and memory)
  * **Effects:** Larger values let the model see longer contexts but increase memory & compute. If `maxlen` is too short, you lose long-distance context and may truncate important data.
  * **Tuning tip:** Set to cover the majority (e.g., 90–95%) of sentence lengths in your data; inspect length histogram.

* `vocab_size` (int)

  * **What it does:** Number of distinct tokens in the vocabulary (including special tokens).
  * **Default:** computed from dataset (e.g., `len(vocab.id2word)`)
  * **Effects:** Larger vocabulary increases embedding table size; using too small vocabulary will map many words to `<UNK>`.
  * **Tuning tip:** Consider frequency thresholding (cut rare words) or use subword tokenization if vocabulary grows too large.

* `num_tags` (int)

  * **What it does:** Number of distinct POS tags (including `<PAD>`).
  * **Default:** computed from dataset
  * **Effects:** Output layer size equals `num_tags`. Larger tag sets require more capacity/data.

### Embedding & Representation

* `embed_dim` (int)

  * **What it does:** Dimensionality of token and position embeddings (the model hidden size).
  * **Default:** `128`
  * **Typical range:** 64–1024 (small tasks: 64–256; larger: 512+)
  * **Effects:** Higher values increase representational capacity and memory/compute. Too small can underfit.
  * **Tuning tip:** Increase when model underfits and you have enough data and compute.

### Transformer-specific

* `num_heads` (int)

  * **What it does:** Number of attention heads in multi-head attention.
  * **Default:** `4`
  * **Typical range:** 1–16 (must divide `embed_dim` if using certain implementations; `key_dim = embed_dim // num_heads`)
  * **Effects:** More heads let the model attend to different subspaces. Increasing heads increases compute (linearly).
  * **Tuning tip:** Keep `embed_dim` divisible by `num_heads`. For small embed sizes, avoid too many heads.

* `ff_dim` (int)

  * **What it does:** Inner dimensionality of the Transformer's position-wise feed-forward network.
  * **Default:** `128`
  * **Typical range:** `embed_dim * 2` to `embed_dim * 4` (commonly)
  * **Effects:** Larger FFN expands model capacity and increases parameters/computation.
  * **Tuning tip:** If FFN is small relative to `embed_dim`, the model may have limited capacity to transform features.

* `num_layers` (int)

  * **What it does:** Number of stacked Transformer blocks.
  * **Default:** `1`
  * **Typical range:** 1–12+ (depending on dataset and budget)
  * **Effects:** More layers allow deeper modeling of hierarchical features — increases capacity & training time.
  * **Tuning tip:** Start small (1–2) for POS tagging; increase only if underfitting.

* `dropout_rate` (float)

  * **What it does:** Dropout probability applied to attention outputs and FFN outputs.
  * **Default in TransformerBlock:** `0.1`
  * **Typical range:** 0.0–0.5
  * **Effects:** Higher dropout improves regularization but too much harms learning.
  * **Tuning tip:** 0.1–0.2 is common for small-medium models.

### Training Hyperparameters

* `lr` (float) — learning rate

  * **What it does:** Step size for the optimizer (how large weight updates are).
  * **Default:** `0.001`
  * **Typical range:** 1e-5 to 1e-2 depending on optimizer and schedule
  * **Effects:** Too large causes instability/divergence; too small slows training and can get stuck.
  * **Tuning tip:** Use warmup schedules or learning-rate decay; try 1e-3 for Adam on small networks, reduce if validation loss oscillates.

* `batch_size` (int)

  * **What it does:** Number of samples processed before updating model weights.
  * **Default used in notebook:** `32` (passed to `fit`)
  * **Typical range:** 8–512 (GPU memory dependent)
  * **Effects:** Larger batches yield more stable gradient estimates and use hardware efficiently but require more memory and may generalize slightly worse. Small batches add noise but may help generalization.
  * **Tuning tip:** Increase while observing GPU memory; use gradient-accumulation if you need effectively larger batches.

* `epochs` (int)

  * **What it does:** Number of full passes over the training set.
  * **Default:** `5` in the notebook
  * **Effects:** More epochs allow more learning but risk overfitting.
  * **Tuning tip:** Use early stopping on validation loss/accuracy.

* `optimizer` and `loss`

  * **What they do:** Optimizer (e.g., Adam) controls weight updates; loss (`sparse_categorical_crossentropy`) computes training signal for multi-class token-level classification.
  * **Default:** Adam with `lr` as configured; `sparse_categorical_crossentropy`.
  * **Tuning tip:** Adam is a good default; consider AdamW or learning-rate schedulers for larger models.

### LSTM-specific

* `lstm_units` (int)

  * **What it does:** Number of hidden units per LSTM layer (per direction if bidirectional).
  * **Default in notebook:** `64` (used inside `Bidirectional(LSTM(64, ...))`)
  * **Effects:** Larger units increase capacity & parameters; too small may underfit.
  * **Tuning tip:** Balance with dataset size; small POS datasets often do fine with 64–256 units.

### Inference & Masking

* `mask_zero=True` (Embedding layer)

  * **What it does:** Treats token id `0` as padding and masks out its embeddings in subsequent layers.
  * **Effects:** Prevents the model from attending to / learning from padded positions. Important for variable-length sequences.
  * **Tuning tip:** Keep `mask_zero=True` if you use `0` for `<PAD>`; some custom layers (e.g., `MultiHeadAttention`) may not automatically handle Keras masks—check compatibility.

---

## Configuration (example)

Use this `config` in the notebook or script. Adjust values to your dataset and compute budget.

```python
config = {
    "maxlen": 30,
    "embed_dim": 128,
    "num_heads": 4,
    "ff_dim": 128,
    "num_layers": 1,
    "lr": 0.001,
    "epochs": 5,
    # these are set after you load the vocab:
    # "vocab_size": len(vocab.id2word),
    # "num_tags": len(vocab.id2tag)
}
batch_size = 32
```

---

## Training & Evaluation Workflow

1. **Load & preprocess data** (e.g., `load_ud`, `load_brown` utilities).
2. **Build vocabulary** and compute `vocab_size`, `num_tags`.
3. **Encode** sentences into `word_ids` & `tag_ids` and `pad_sequences` to `maxlen`.
4. **Split** into train/val/test (e.g., `train_test_split`).
5. **Build models**:

   * `transformer_model = build_model("transformer", config)`
   * `lstm_model = build_model("lstm", config)`
6. **Train** with `model.fit(X_train, y_train, validation_split=0.2, epochs=..., batch_size=...)`
7. **Evaluate** on test set: token-level metrics, sentence-level accuracy, per-tag accuracy. Use utilities provided: `evaluate_models_on_test_set`, `plot_confusion_matrix`, `plot_attention_map`, etc.
8. **Visualize** training curves and attention to interpret behavior.

---

## Visualization & Analysis Tools

* `plot_history(history, title)` – training / validation accuracy curves
* `predict_example(model, vocab, raw_sentences, encoded_data, maxlen, index)` – prints token-level predictions vs ground truth
* `plot_confusion_matrix(model, X_test, y_test, vocab)` – shows per-tag confusion matrix (padding ignored)
* `plot_attention_map(model, vocab, raw_sentences, encoded_data, maxlen, index)` – renders average attention heatmap for a sentence
* `evaluate_models_on_test_set(transformer_model, lstm_model, X_test, y_test, vocab)` – comprehensive test set comparison

---

## Data Loaders

This project includes helper loaders:

* `load_ud(split="train", n=1000)` — loads `n` samples from Universal Dependencies `en_ewt` and returns `(preprocessed, vocab, encoded)`.
* `load_brown()` — downloads and preprocesses the Brown corpus (with tag mapping to UPOS) and returns `(preprocessed, vocab, encoded)`.

These utilities normalize tokens and tags and build the `Vocabulary` dataclass that provides `encode()` and `decode()` helpers.

---

## Practical Tips & Troubleshooting

* If validation accuracy is unstable:

  * Reduce `lr`.
  * Use smaller `batch_size` or gradient clipping.
  * Add a learning rate schedule or warmup.
* If model underfits:

  * Increase `embed_dim`, `ff_dim`, or `num_layers`.
  * Add more training data or augment with external corpora.
* If model overfits:

  * Increase dropout, reduce model size, add weight decay, or use early stopping.
* If many `<UNK>` tokens:

  * Increase vocabulary coverage, use subword tokenizers (BPE / WordPiece) or pretrained embeddings.
* Attention map returns noisy patterns if `mask` handling is inconsistent — confirm mask propagation through custom layers.

---

## Technologies Used

This project uses the following tools & libraries:

* TensorFlow / Keras for model building and training
* NumPy for numerical ops
* NLTK for corpus utilities (Brown)
* Hugging Face Datasets for Universal Dependencies loader
* Matplotlib and Seaborn for visualizations
* scikit-learn for metrics & utilities
* Brown Corpus (data): Brown Corpus
* Universal Dependencies subset: Universal Dependencies

---

## Example: building & training (snippet)

```python
# after loading vocab and encoded data
config["vocab_size"] = len(vocab.id2word)
config["num_tags"] = len(vocab.id2tag)

transformer = build_model("transformer", config)
lstm = build_model("lstm", config)

h_trans = transformer.fit(X_train, y_train, batch_size=32, epochs=config["epochs"], validation_split=0.2)
h_lstm = lstm.fit(X_train, y_train, batch_size=32, epochs=config["epochs"], validation_split=0.2)
```

---

## Future Improvements

* Add subword tokenization (BPE / WordPiece) to reduce `<UNK>` rate and learn richer morphological patterns.
* Integrate pretrained contextual embeddings (e.g., a frozen or fine-tuned transformer encoder) to bootstrap performance.
* Add more Transformer layers and/or layer normalization variants (Pre-LN vs Post-LN) for stability.
* Implement learning-rate schedules (warmup + cosine decay) and AdamW for better generalization.
* Expand visualization tooling (per-head attention attribution, integrated gradients, interactive dashboards).

---

## License & Contribution

* Add your license of choice (e.g., MIT).
* Contributions welcome: open issues for bugs, feature requests (subword tokenization, pretrained embedding support, more datasets).

---

## Contact / Notes

* This README is intended to be a complete project overview plus an explanation of the hyperparameters you can tune.
* For reproducible experiments, always log model configuration (all hyperparameters), random seeds, and dataset splits.
