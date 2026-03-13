from logging import config

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import confusion_matrix
import seaborn as sns

import matplotlib.pyplot as plt

from data.datasetsss import load_ud

from data.vocabulary import Encoding

# --- DATA INFRASTRUCTURE ---

def prepare_for_keras(encoded_list, maxlen):
    """Turns the list of Encoding dataclasses into clean NumPy matrices."""
    word_ids = [e.word_ids for e in encoded_list]
    tag_ids = [e.tag_ids for e in encoded_list]
    
    # We pad with 0 (which matches our <PAD> token)
    X = pad_sequences(word_ids, maxlen=maxlen, padding='post', value=0)
    y = pad_sequences(tag_ids, maxlen=maxlen, padding='post', value=0)
    return np.array(X), np.array(y)

# --- ARCHITECTURE COMPONENTS ---

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"), 
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        # Attention + Residual + Norm
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # FFN + Residual + Norm
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# --- EXPERIMENT RUNNER ---

def build_model(model_type, config):
    inputs = layers.Input(shape=(config['maxlen'],))
    
    if model_type == "transformer":
        x = TokenAndPositionEmbedding(config['maxlen'], config['vocab_size'], config['embed_dim'])(inputs)
        for _ in range(config.get('num_layers', 1)):
            x = TransformerBlock(config['embed_dim'], config['num_heads'], config['ff_dim'])(x)
    elif model_type == "lstm":
        x = layers.Embedding(config['vocab_size'], config['embed_dim'], mask_zero=True)(inputs)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    
    outputs = layers.Dense(config['num_tags'], activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=keras.optimizers.Adam(config['lr']),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# --- EVALUATION & VISUALIZATION ---

def plot_history(history, title):
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'Accuracy for {title}')
    plt.legend()
    plt.show()

def predict_example(model, vocab, raw_sentences, encoded_data, maxlen, index=0):
    tokens, true_tags = raw_sentences[index]
    input_padded = pad_sequences([encoded_data[index].word_ids], maxlen=maxlen, padding='post')
    
    preds = model.predict(input_padded)
    # Get highest probability tag for each word, skip padding
    pred_indices = np.argmax(preds[0], axis=-1)[:len(tokens)]
    pred_tags = [vocab.id2tag[idx] for idx in pred_indices]
    
    print(f"\nExample Sentence: {' '.join(tokens)}")
    print(f"{'WORD':<15} | {'TRUE':<10} | {'PRED'}")
    print("-" * 40)
    for w, t, p in zip(tokens, true_tags, pred_tags):
        print(f"{w:<15} | {t:<10} | {p} {'✅' if t==p else '❌'}")

def plot_confusion_matrix(model, X_test, y_test, vocab):
    # 1. Get predictions
    preds = model.predict(X_test)
    y_pred = np.argmax(preds, axis=-1).flatten()
    y_true = y_test.flatten()
    
    # 2. Filter out the padding (ID 0) so it doesn't skew the results
    mask = y_true != 0
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    # 3. Create Matrix
    cm = confusion_matrix(y_true_filtered, y_pred_filtered)
    # Get the names of the tags for the axes
    tag_names = vocab.id2tag
    
    # 4. Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=tag_names, yticklabels=tag_names, cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('POS Tag Confusion Matrix')
    plt.show()

def plot_attention_map(model, vocab, raw_sentences, encoded_data, maxlen, index=0):
    tokens, _ = raw_sentences[index]
    input_padded = pad_sequences([encoded_data[index].word_ids], maxlen=maxlen, padding='post')
    
    # We need to create a sub-model that outputs the attention weights
    # This assumes your TransformerBlock has an attribute called 'att'
    # and we capture the weights during the 'call'
    
    # This is a bit advanced: We access the 'att' layer of the transformer block
    # and get its output weights.
    attn_layer = model.layers[2] # Adjust index based on your model.summary()
    
    # We use the Keras backend to get the weights for a specific input
    # Or more simply, re-run a custom predict function:
    _, weights = attn_layer.att(model.layers[1](input_padded), 
                                model.layers[1](input_padded), 
                                return_attention_scores=True)
    
    # Average across all heads for a global view
    avg_weights = np.mean(weights[0], axis=0)[:len(tokens), :len(tokens)]

    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_weights, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
    plt.title(f"Attention Map: {' '.join(tokens)}")
    plt.show()

def compare_histories(trans_history, lstm_history):
    plt.figure(figsize=(12, 5))
    
    # Accuracy Comparison
    plt.subplot(1, 2, 1)
    plt.plot(trans_history.history['val_accuracy'], label='Transformer Val Acc', color='blue')
    plt.plot(lstm_history.history['val_accuracy'], label='LSTM Val Acc', color='orange', linestyle='--')
    plt.title('Accuracy: Transformer vs LSTM')
    plt.legend()
    
    # Loss Comparison
    plt.subplot(1, 2, 2)
    plt.plot(trans_history.history['val_loss'], label='Transformer Val Loss', color='blue')
    plt.plot(lstm_history.history['val_loss'], label='LSTM Val Loss', color='orange', linestyle='--')
    plt.title('Loss: Transformer vs LSTM')
    plt.legend()
    
    plt.show()

def main() -> None:
    # Configuration Dictionary
    MAX_LEN = 30
    config = {
        "maxlen": MAX_LEN,
        "embed_dim": 128,
        "num_heads": 4,
        "ff_dim": 128,
        "num_layers": 1,
        "lr": 0.001,
        "epochs": 5
    }

    # Load Data
    raw_data, vocab, encoded = load_ud(n=2000)
    X, y = prepare_for_keras(encoded, MAX_LEN)

    config["vocab_size"] = len(vocab.id2word)
    config["num_tags"] = len(vocab.id2tag)

    # Train Transformer
    transformer_model = build_model("transformer", config)
    h_trans = transformer_model.fit(X, y, batch_size=32, epochs=config['epochs'], validation_split=0.2)

    # Train LSTM for Comparison
    lstm_model = build_model("lstm", config)
    h_lstm = lstm_model.fit(X, y, batch_size=32, epochs=config['epochs'], validation_split=0.2)

    # Visualize Results
    # Compare training curves
    compare_histories(h_trans, h_lstm)
    
    # Show a specific example
    predict_example(transformer_model, vocab, raw_data, encoded, MAX_LEN, index=15)
    
    # Generate the Heatmap for the same example
    plot_attention_map(transformer_model, vocab, raw_data, encoded, MAX_LEN, index=15)
    
    # Plot the Confusion Matrix
    plot_confusion_matrix(transformer_model, X, y, vocab)

    
if __name__ == "__main__":
    main()
