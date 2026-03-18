# Hypotheses Reference

This document defines each hypothesis as a testable statement with clear criteria.

## Global Evaluation Protocol

- Primary metric: token accuracy on the held-out test split from app-level split.
- Secondary metrics: macro-F1, weighted-F1, training stability (std across seeds when applicable).
- Statistical reporting: report mean and std across seeds when running multiple seeds.
- Decision rule template: prefer differences that are both practically meaningful and consistent.

## H1: Learning Rate Preference by Architecture

### Research question
Do transformer and LSTM architectures have different optimal learning rate ranges under comparable capacity?

### Null hypothesis (H0)
For fixed architecture size and sentence budget, changing learning rate does not produce a meaningful performance difference.

### Alternative hypothesis (H1)
Transformer models perform best at lower learning rates than LSTM models under the same training protocol.

### Experimental factors
- Manipulated: `lr` in `{1e-5, 1e-4, 1e-3}`.
- Controlled:
	- Model family pair: medium transformer vs stronger LSTM.
	- Sentence range: 5000 to 13000, step 2000.
	- `maxlen=30`, fixed split seed and run seed per config.

### Success criteria
- For transformer runs, best mean token accuracy should occur at `1e-5` or `1e-4`.
- For LSTM runs, best mean token accuracy should occur at `1e-4` or `1e-3`.
- Prefer at least 1.0 absolute token-accuracy point between best and worst LR in each family.

## H2: Small Models Are More LR-Sensitive

### Research question
Are smaller models more sensitive to learning-rate changes than larger models?

### Null hypothesis (H0)
Learning-rate sensitivity is equivalent between small and medium-capacity variants.

### Alternative hypothesis (H2)
Small models show larger performance variance across learning rates than medium models.

### Experimental factors
- Manipulated:
	- Model capacity: small vs medium.
	- Learning rate: `{1e-5, 1e-4, 1e-3}`.
- Controlled:
	- Sentence range: 5000 to 13000, step 2000.
	- Same optimizer family and training pipeline.

### Success criteria
- Define LR sensitivity per architecture as: max(token_accuracy over LR) - min(token_accuracy over LR).
- H2 is supported if small-model sensitivity > medium-model sensitivity for both transformer and LSTM families.

## H3: Best-Found Transformer vs Best-Found LSTM

### Research question
Using best-found LR per architecture, how does performance scale with more real Brown data?

### Null hypothesis (H0)
At equal sentence budgets, there is no meaningful performance difference between best transformer and best LSTM.

### Alternative hypothesis (H3)
Transformer improves faster or reaches a higher asymptote than LSTM as sentence budget increases.

### Experimental factors
- Dataset: Brown.
- Sentence sweep: 10000 to 57000, step 12000.
- Fixed best LR per family from H1/H2 outcomes.

### Success criteria
- Compare curve slope and final-point score.
- H3 is supported if transformer is consistently >= LSTM across most sentence points and ends higher at largest non-resampled point.