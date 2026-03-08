import argparse
# from pipeline.runner import run

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from data.datasetsss import load_ud

from data.vocabulary import Encoding

def main() -> None:
    parser = argparse.ArgumentParser(description="POS Tagging Pipeline")
    parser.add_argument(
        "--steps",
        nargs="+",
        metavar="STEP",
        help="Run only these steps (e.g. step_01_data step_02_embedding)",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        metavar="STEP",
        help="Skip these steps",
    )
    args = parser.parse_args()

    # run(steps_filter=args.steps, skip_filter=args.skip)

    raw_sentences, vocab, encoded_data = load_ud(n=5000)

    maxlen = 30         # Max length of a sentence
    embed_dim = 128     # Embedding size
    num_heads = 4       # Number of attention "heads"
    ff_dim = 128        # Hidden layer size in feed forward

    vocab_size = len(vocab.id2word)
    num_tags = len(vocab.id2tag)

    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    X = embedding_layer(inputs)

    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    X = transformer_block(X)

    # Final Output Layer: One prediction for EVERY word in the sequence
    outputs = layers.Dense(num_tags, activation="softmax")(X)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    X_train, y_train = prepare_for_keras(encoded_data, maxlen=maxlen)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    check_model_performance(model, vocab, raw_sentences, encoded_data, maxlen, index=42) 
    

def prepare_for_keras(encoded_list: list[Encoding], maxlen: int):
    # 1. Extract the lists from the dataclasses
    word_ids = [e.word_ids for e in encoded_list]
    tag_ids = [e.tag_ids for e in encoded_list]

    # 2. PAD them so every sentence is exactly 'maxlen' long
    # We use padding='post' so the zeros go at the end of the sentence
    X = pad_sequences(word_ids, maxlen=maxlen, padding='post', value=0)
    y = pad_sequences(tag_ids, maxlen=maxlen, padding='post', value=0)

    return np.array(X), np.array(y)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero = True)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim, mask_zero = True)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions # Combine them

class TransformerBlock(layers.Layer):
    #word vector size, heads, number of neurons, dropout rate
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()

        # 1. Multi-Head Attention (The Context Layer)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # Query/Key/Value happens here ^

        # 2. NN
        self.fnn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"), 
            layers.Dense(embed_dim)
        ])

        # 3. Normalization Layers
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()

        # 4. Dropout
        self.dropout1 = layers.Dropout(rate)

    def call(self, inputs, training = False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out = self.layernorm1(inputs + attn_output) # The "ADD & NORM"

        ffn_output = self.fnn(out)
        return self.layernorm2(out + ffn_output)

import numpy as np

def check_model_performance(model, vocab, raw_sentences, encoded_data, maxlen, index=0):
    # 1. Get the Raw Sentence and True Tags
    tokens, true_tags = raw_sentences[index]
    
    # 2. Get the Prepared Input (already padded/indexed by Person A)
    # encoded_data[index].word_ids is what we need
    input_data = np.array([encoded_data[index].word_ids])
    
    # Ensure it's padded correctly if it wasn't already
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    input_padded = pad_sequences(input_data, maxlen=maxlen, padding='post')

    # 3. Make the Prediction
    # This returns probabilities for every tag for every word
    predictions = model.predict(input_padded) 
    
    # 4. Decode the Prediction
    # We take the 'argmax' (the tag with the highest probability)
    # We only look at the first 'len(tokens)' words to ignore padding
    predicted_indices = np.argmax(predictions[0], axis=-1)[:len(tokens)]
    predicted_tags = [vocab.id2tag[idx] for idx in predicted_indices]

    # 5. Print a pretty comparison
    print(f"{'WORD':<15} | {'TRUE TAG':<10} | {'PREDICTED'}")
    print("-" * 40)
    for word, true, pred in zip(tokens, true_tags, predicted_tags):
        status = "✅" if true == pred else "❌"
        print(f"{word:<15} | {true:<10} | {pred} {status}")

if __name__ == "__main__":
    main()
