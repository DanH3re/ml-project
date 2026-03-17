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

## Notes

- Hypothesis IDs come from the `hypothesis` field in each config JSON.
- Description text is taken from the Group A-G markdown sections in src/pos_hypothesis_explorer.ipynb.
