# Hypotheses Reference

## H1: Transformers prefer lower learning rates and scale better with data

**Hypothesis:** Transformers achieve their best performance at lower learning rates than LSTMs. As training data size increases, Transformers significantly outperform LSTMs, even at learning rates that hurt LSTM performance.

**Expected pattern:**
- LSTM peaks at higher learning rates; Transformer peaks at lower learning rates
- Transformer's advantage grows with more training data
- At high data regimes (10K+ sentences), Transformer should substantially outperform LSTM

## H2: Small Transformers outperform small LSTMs at comparable capacity

**Hypothesis:** For similar model capacity (roughly matched parameter count) and the same training setup, a small Transformer achieves better validation performance than a small LSTM.

**Expected pattern:**
- With matched capacity and training budget, Transformer reaches higher peak validation score than LSTM
- The advantage is most visible once there is enough data to train attention effectively (i.e., not only in the tiniest data regime)
- Results are robust across a reasonable hyperparameter sweep (the best Transformer run beats the best LSTM run)

## H3: At a fixed low learning rate, Transformers benefit more from more data

**Hypothesis:** With a fixed low learning rate ($\mathrm{lr}=10^{-4}$), Transformer performance improves more steeply than LSTM performance as the number of training sentences increases, eventually outperforming the LSTM.

**Expected pattern:**
- At small data sizes, Transformer and LSTM are similar (or LSTM may be slightly better)
- As sentences increase, Transformer improves faster than LSTM
- The performance gap widens in higher-data regimes (e.g., 10K–40K sentences)

## H4: Transformers are more robust to longer sequences at a fixed learning rate

**Hypothesis:** With $\mathrm{lr}=10^{-4}$ and moderate-to-large training sets (10K–40K sentences), increasing the maximum sequence length hurts the LSTM more than the Transformer; the Transformer remains comparatively stable or degrades more slowly.

**Expected pattern:**
- As max length increases, LSTM performance drops sooner/more sharply than Transformer
- Transformer performance is more stable across max length settings
- Training cost increases for both as max length increases (potentially limiting feasible max length)