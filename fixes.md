**Fix 1. The attention mask was missing (critical bug)**

In `train_pos.py:98`, the transformer's `MultiHeadAttention` call doesn't pass the padding mask:
```python
attn_output = self.att(inputs, inputs)  # no attention_mask!
```
The LSTM gets mask propagation for free — Keras automatically passes the mask from `Embedding(mask_zero=True)` through `Bidirectional(LSTM(...))`. The transformer attends to padding tokens on every layer, polluting its representations. This is the single biggest reason.

**2. Val accuracy vs test accuracy gap reveals the problem**

Look at the val vs test accuracy gap by max_len (Group C):

| model | len | best_val_acc | test_acc | gap |
|---|---|---|---|---|
| transformer | 20 | 0.871 | 0.816 | **-5.5%** |
| lstm | 20 | 0.875 | 0.876 | +0.1% |
| transformer | 100 | **0.968** | 0.834 | **-13.4%** |
| lstm | 100 | 0.894 | 0.898 | +0.4% |

The transformer gap *grows* with longer max_len. More padding → worse attention → bigger gap. The LSTM's val/test are nearly identical throughout. This is the signature of the broken mask.

**3. Dataset is too small for transformers**

2000 sentences is very little. Transformers need more data to learn useful attention patterns. LSTM's sequential inductive bias is a much better fit for POS tagging on small datasets.

**4. What to do about it**

To fix the transformer, pass the mask from the embedding to the attention layer in `TransformerBlock.call`:
```python
def call(self, inputs, training=False):
    # Get the mask from the embedding (propagated automatically)
    mask = self._compute_causal_mask(inputs) if hasattr(self, '_mask') else None
```

Actually the proper fix is to pass `attention_mask` to `self.att(...)` using the boolean mask from the embedding:

```python
attn_output = self.att(inputs, inputs, attention_mask=some_mask)
```

The embedding layer's mask is available via `supports_masking = True` and is propagated through the Keras functional API, but the `TransformerBlock.call` never uses it. You'd need to either accept a `mask` argument or extract it from the layer's `_keras_mask` attribute.