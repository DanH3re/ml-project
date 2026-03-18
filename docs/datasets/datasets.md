# Datasets for POS Tagging

This document describes the datasets currently supported by this project.

## Currently Implemented Datasets

### 1. Universal Dependencies - English EWT (ud)
- **Description**: The English Web Treebank (EWT) from Universal Dependencies. Contains text from blogs, newsgroups, emails, reviews, and Yahoo! Answers.
- **Size**: ~16,600 sentences across train/validation/test splits
- **Tag Set**: Universal POS tags (UPOS)
- **Genre**: Web text (diverse, informal)
- **Best For**: Modern, informal text; split-based experiments
- **Usage**: `"dataset": "ud"`
- **Recommended maxlen**: 30-40
- **Notes**:
  - Supports `split="train" | "validation" | "test"` in loader-level use.
  - In config-based training, project-level train/test split is still applied by the pipeline.

### 2. Brown Corpus (brown)
- **Description**: Classic balanced corpus from 1961 containing American English text across multiple genres.
- **Size**: ~57,000 sentences
- **Tag Set**: Brown tags preprocessed through the project vocabulary/tag pipeline
- **Genre**: News, fiction, government documents, academic prose, etc.
- **Best For**: Larger corpus runs and baseline comparisons
- **Usage**: `"dataset": "brown"`
- **Recommended maxlen**: 25-35
- **Notes**:
  - Exposed as a single corpus (`all`) at loader level.
  - Config-based runs must use an explicit integer for `"sentences"`.
  - If requested sentences exceed dataset capacity, training now raises a validation error.

## Dataset Comparison Table

| Dataset | Size | Split Support | Domain | Best Use Case |
|---------|------|---------------|--------|---------------|
| UD EWT | ~16.6k | train/validation/test | Web/General | Modern text, split experiments |
| Brown | ~57k | all | Multi-genre | Larger-scale baseline training |

## Usage Examples

### Basic Usage in Config Files

```json
{
  "name": "experiment_name",
  "dataset": "brown",
  "sentences": 2000,
  "maxlen": 30
}
```

### Large Brown Run

```json
{
  "dataset": "brown",
  "sentences": 30000,
  "maxlen": 30
}
```

### Dataset-Specific Recommendations

#### Quick Iteration

```json
{
  "dataset": "ud",
  "sentences": 2000,
  "maxlen": 30
}
```

#### Larger Runs

```json
{
  "dataset": "brown",
  "sentences": 30000,
  "maxlen": 30
}
```

## Experimental Design Recommendations

### H1: Architecture Comparison
- Use one fixed dataset for all runs (for example, UD or Brown).

### H2: Data Efficiency Curves
- Keep architecture fixed and vary `"sentences"` with explicit integers, for example: `[100, 500, 1000, 2000, 5000]`.

### H3: Split Sensitivity (UD)
- Evaluate behavior across `train`, `validation`, and `test` at loader-level analysis.

## Running Experiments

### Single Config

```bash
python src/train_pos.py --config resources/configs/hypothesis_3_best_lr_brown.json
```

### All Configs

```bash
python src/run_all_configs.py --configs-dir resources/configs --use-gpu amd --continue-on-error
```

## Notes

- All loaders normalize tokens using the project preprocessing pipeline.
- Special tokens: `<PAD>` (id=0) and `<UNK>` for unknown words.
- Dataset caching is implemented; repeated use of the same dataset/sentence count is loaded once.

## References

- Universal Dependencies: https://universaldependencies.org/
- Brown Corpus: Francis & Kucera (1979)
