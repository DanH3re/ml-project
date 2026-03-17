# Hypotheses Reference

Generated: 2026-03-17

| Group | Hypothesis ID | Description | Primary source |
|---|---|---|---|
| A | H1_transformer_beats_lstm_default | A medium Transformer should outperform a reasonable BiLSTM on the default setup. | resources/configs/group_A_baseline_family_comparison.json |
| B | H2_lstm_more_competitive_low_data | LSTM is more competitive on small data. | resources/configs/group_B_data_regime.json |
| B | H4_transformer_scales_better_with_more_data | Transformer scales better as data increases. | resources/configs/group_B_data_regime.json |
| C | H3_transformer_gains_more_from_longer_context | Transformer should gain more from longer available context than LSTM. | resources/configs/group_C_context_length_regime.json |
| D | H7_transformer_more_sensitive_to_learning_rate | Transformer is more sensitive to learning rate. | resources/configs/group_D_optimization_sensitivity.json |
| D | H8_transformer_more_sensitive_to_dropout | Transformer is more sensitive to dropout. | resources/configs/group_D_optimization_sensitivity.json |
| D | H9_batch_size_affects_transformer_more | Batch size affects Transformer more strongly than LSTM. | resources/configs/group_D_optimization_sensitivity.json |
| E | H10_results_not_stable_across_seeds | Some apparent wins may be unstable and depend on lucky data splits or training randomness. | resources/configs/group_E_robustness.json |
| F | H11_warmup_improves_transformer | Linear LR warmup should stabilise Transformer training and improve final accuracy; LSTM is largely unaffected because it converges more smoothly from step one. | resources/configs/group_F_lr_warmup.json |
| G | H12_depth_helps_architecture | Adding more layers (depth) helps both families, but the effect is stronger for transformers. | resources/configs/group_G_architecture_depth.json |

NEW!!!

## H1: Transformers prefer lower learning rates and scale better with data

**Hypothesis:** Transformers achieve their best performance at lower learning rates than LSTMs. As training data size increases, Transformers significantly outperform LSTMs, even at learning rates that hurt LSTM performance.

**Key comparisons:**
- **Group B (Data Regime)**: Transformer vs LSTM performance as sentence count increases (500 → 2000 → 5000 → 12543)
- **Group D (Learning Rate Sensitivity)**: F1 scores across learning rates (1e-4 to 1e-2) for both architectures

**Expected pattern:**
- LSTM peaks at higher learning rates; Transformer peaks at lower learning rates
- Transformer's advantage grows with more training data
- At high data regimes (10K+ sentences), Transformer should substantially outperform LSTM