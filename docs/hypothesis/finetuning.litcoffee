# Transformer

```json
{
  "sentences": 13000,
  "maxlen": 30,
  "num_layers": [1, 2, 3],
  "embed_dim": [64, 128, 192, 256],
  "num_heads": [2, 4, 8],
  "ff_dim": [256, 512, 1024],
  "dropout": [0.0, 0.1, 0.2, 0.3],
  "lr": [1e-5, 2e-5, 3e-5, 5e-5],
  "batch_size": [16, 32, 64],
  "epochs": 100,
  "early_stopping": true,
  "patience": 2
}
```

Suggested values – sources:

- num_layers: [1, 2, 3]  
  https://www.cambridge.org/core/journals/natural-language-processing/article/partofspeech-tagger-for-bodo-language-using-deep-learning-approach/8E223194F0DCB9837640577190631CC3  

- embed_dim: [64, 128, 256]  
  https://buomsoo-kim.github.io/attention/2020/04/22/Attention-mechanism-20.md/  

- num_heads: [2, 4, 8]  
  https://buomsoo-kim.github.io/attention/2020/04/22/Attention-mechanism-20.md/  

- ff_dim: [256, 512, 1024]  
  https://buomsoo-kim.github.io/attention/2020/04/22/Attention-mechanism-20.md/  

- dropout: [0.0, 0.1, 0.2, 0.3]  
  https://buomsoo-kim.github.io/attention/2020/04/22/Attention-mechanism-20.md/  

- lr: [1e-5, 2e-5, 3e-5, 5e-5]  
  https://github.com/pjlintw/Transformer-py  
  https://arxiv.org/html/2510.13854v1  

- batch_size: [16, 32, 64]  
  https://github.com/pjlintw/Transformer-py  
  https://wandb.ai/renaudlesperance/INF8225%20-%20TP3%20-%20Final%20Run/reports/Transformer-Hyperparameter-Tuning--VmlldzoxODEyNDQ

# BiLSTM

```json
{
    "sentences": 13000,
    "maxlen": 30,
    "embed_dim": [64, 128, 192],
    "lstm_units": [64, 160, 256],
    "dropout": [0.1, 0.2, 0.3],
    "lr": [0.001, 0.003, 0.005],
    "epochs": 100,
    "batch_size": [8, 16, 32],
    "early_stopping": true,
    "patience": 2
}
```

**Suggested Values – Sources**

- **embed_dim**: [64, 128, 192]  
  https://public.ukp.informatik.tu-darmstadt.de/reimers/Optimal_Hyperparameters_for_Deep_LSTM-Networks.pdf
- **lstm_units**: [64, 160, 256]  
  https://public.ukp.informatik.tu-darmstadt.de/reimers/Optimal_Hyperparameters_for_Deep_LSTM-Networks.pdf
- **dropout**: [0.1, 0.2, 0.3]  
  https://ar5iv.labs.arxiv.org/html/1707.06799
- **lr**: [0.001, 0.003, 0.005]  
  https://public.ukp.informatik.tu-darmstadt.de/reimers/Optimal_Hyperparameters_for_Deep_LSTM-Networks.pdf
- **batch_size**: [8, 16, 32]  
  https://public.ukp.informatik.tu-darmstadt.de/reimers/Optimal_Hyperparameters_for_Deep_LSTM-Networks.pdf