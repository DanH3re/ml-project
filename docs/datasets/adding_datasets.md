# How to Add a New Dataset

This guide explains how to add support for a new POS tagging dataset to the project.

## Quick Steps

1. **Add a loader function in `src/data/dataset.py`**
2. **Register it in `load_dataset_by_name()`**
3. **Create config files in `resources/configs/<dataset_name>/`**
4. **Test it with `test_datasets.py`**
5. **Document it in `docs/datasets.md`**

## Detailed Instructions

### Step 1: Create a Loader Function

Add a new function to `src/data/dataset.py`:

```python
def load_your_dataset(split: str = "train", n: int | None = 1000) -> tuple[list[Sentence], Vocabulary, list[Encoding]]:
    """
    Load Your Dataset.

    Args:
        split: Data split (train/validation/test) if applicable
        n: Number of sentences to load. None = all sentences.

    Returns:
        (preprocessed_sentences, vocabulary, encoded_data)
    """
    # 1. Load the raw data
    # If using HuggingFace datasets:
    from datasets import load_dataset
    dataset = load_dataset("organization/dataset_name", "config_name")

    # If using NLTK:
    # import nltk
    # nltk.download("dataset_name", quiet=True)
    # from nltk.corpus import dataset_name
    # sentences = dataset_name.tagged_sents()

    # 2. Limit to n sentences if specified
    if dataset_name_from_hf:
        split_data = dataset[split]
        if n is not None:
            n = min(n, len(split_data))
        items = split_data if n is None else split_data.select(range(n))
    else:
        sentences = sentences[:n] if n is not None else sentences

    # 3. Preprocess into (tokens, tags) format
    preprocessed = []
    for item in items:  # or sentence in sentences
        # Extract tokens and tags from your data format
        tokens = preprocess_tokens(item["tokens"])  # or however they're stored
        tags = [tag_names[i] for i in item["tags"]]  # convert to strings
        preprocessed.append((tokens, tags))

    # 4. Build vocabulary and encode
    vocab = build_vocab(preprocessed)
    encoded = [vocab.encode(tokens, tags) for tokens, tags in preprocessed]

    return preprocessed, vocab, encoded
```

### Step 2: Register in load_dataset_by_name

Add your loader to the `loaders` dictionary in `load_dataset_by_name()`:

```python
def load_dataset_by_name(
    dataset_name: str,
    split: str = "train",
    n: int | None = 1000
) -> tuple[list[Sentence], Vocabulary, list[Encoding]]:
    loaders = {
        "ud": lambda: load_ud(split=split, n=n),
        "brown": lambda: load_brown(),
        "your_dataset": lambda: load_your_dataset(split=split, n=n),  # ADD THIS LINE
        # ... other loaders
    }

    if dataset_name not in loaders:
        available = ", ".join(loaders.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")

    return loaders[dataset_name]()
```

### Step 3: Create Config Files

Create a directory and baseline config:

```bash
mkdir -p resources/configs/your_dataset
```

Create `resources/configs/your_dataset/baseline_comparison.json`:

```json
[
  {
    "name": "your_dataset_transformer_small",
    "group": "baseline_comparison",
    "dataset": "your_dataset",
    "model_type": "transformer",
    "sentences": 2000,
    "maxlen": 30,
    "split_seed": 42,
    "seed": 42,
    "embed_dim": 128,
    "num_heads": 4,
    "ff_dim": 128,
    "num_layers": 1,
    "dropout": 0.1,
    "lr": 0.001,
    "epochs": 12,
    "batch_size": 64,
    "early_stopping": true,
    "patience": 3,
    "validation_split": 0.2
  },
  {
    "name": "your_dataset_lstm_baseline",
    "group": "baseline_comparison",
    "dataset": "your_dataset",
    "model_type": "lstm",
    "sentences": 2000,
    "maxlen": 30,
    "split_seed": 42,
    "seed": 42,
    "embed_dim": 128,
    "lstm_units": 64,
    "lr": 0.001,
    "epochs": 15,
    "batch_size": 32,
    "early_stopping": true,
    "patience": 3,
    "validation_split": 0.2
  }
]
```

**Config Parameters to Adjust**:
- `maxlen`: Check average sentence length in your dataset
- `sentences`: Number of sentences (explicit positive integer)
- `epochs`, `batch_size`: Adjust based on dataset size

### Step 4: Test Your Dataset

Run the test script:

```bash
python test_datasets.py
```

Or test just yours:

```python
from src.data.dataset import load_dataset_by_name

raw_data, vocab, encoded = load_dataset_by_name("your_dataset", n=100)
print(f"Loaded {len(raw_data)} sentences")
print(f"Vocab size: {len(vocab.id2word)}")
print(f"Tags: {list(vocab.id2tag.values())}")
```

### Step 5: Update Documentation

Add your dataset to `docs/datasets.md`:

```markdown
### N. Your Dataset Name (your_dataset)
- **Description**: Brief description of the dataset
- **Size**: Number of sentences
- **Tag Set**: Tag set name and number of tags
- **Genre**: Text genre/domain
- **Best For**: Use cases
- **Usage**: `"dataset": "your_dataset"`
- **Recommended maxlen**: Recommended value based on sentence lengths
```

## Common Data Formats

### HuggingFace Datasets

```python
from datasets import load_dataset

# Universal Dependencies format
dataset = load_dataset("universal_dependencies", "en_ewt")
tokens = item["tokens"]
tags = [label_names[i] for i in item["upos"]]

# Generic format
dataset = load_dataset("org/dataset")
tokens = item["words"]  # or "tokens", check schema
tags = item["pos_tags"]  # check schema
```

### NLTK Corpora

```python
import nltk
nltk.download("corpus_name", quiet=True)
from nltk.corpus import corpus_name

# Tagged sentences format
sentences = corpus_name.tagged_sents()
for sent in sentences:
    tokens, tags = zip(*sent)
```

### Custom Files (CoNLL format)

```python
def load_conll_file(file_path: str):
    sentences = []
    current_tokens = []
    current_tags = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # Empty line = sentence boundary
                if current_tokens:
                    sentences.append((current_tokens, current_tags))
                    current_tokens = []
                    current_tags = []
            else:
                parts = line.split('\t')  # or split()
                token = parts[0]
                tag = parts[1]  # adjust index based on format
                current_tokens.append(token)
                current_tags.append(tag)

    return sentences
```

## Tips

### Determining maxlen

Check your dataset's sentence length distribution:

```python
import numpy as np
raw_data, vocab, encoded = load_dataset_by_name("your_dataset", n=None)
lengths = [len(tokens) for tokens, tags in raw_data]
print(f"Mean: {np.mean(lengths):.1f}")
print(f"Median: {np.median(lengths):.1f}")
print(f"95th percentile: {np.percentile(lengths, 95):.1f}")
print(f"Max: {np.max(lengths)}")

# Set maxlen to cover ~95% of sentences
```

### Tag Set Normalization

If your dataset has a non-standard tag set, you may want to map it:

```python
TAG_MAP = {
    "NNS": "NOUN",
    "VBZ": "VERB",
    # ... more mappings
}

tags = [TAG_MAP.get(tag, tag) for tag in original_tags]
```

Or use it as-is for fine-grained experiments.

### Handling Missing Data

```python
# Filter out sentences with missing tags
preprocessed = [
    (tokens, tags)
    for tokens, tags in raw_data
    if len(tokens) == len(tags) and all(tags)
]
```

## Running Experiments

Once your dataset is added:

```bash
# Single config
python src/train_pos.py \
  --config resources/configs/your_dataset/baseline_comparison.json \
  --use-gpu amd

# All configs for your dataset
python src/run_all_configs.py \
  --configs-dir resources/configs/your_dataset \
  --use-gpu amd
```

## Example: Adding OntoNotes

Here's a complete example for OntoNotes:

```python
def load_ontonotes(split: str = "train", n: int | None = 1000) -> tuple[list[Sentence], Vocabulary, list[Encoding]]:
    """Load OntoNotes 5.0 English corpus."""
    # This is pseudocode - actual implementation depends on OntoNotes format
    dataset = load_dataset("ontonotes5", "english_v12")

    split_data = dataset[split]
    if n is not None:
        n = min(n, len(split_data))
    items = split_data if n is None else split_data.select(range(n))

    preprocessed = []
    for item in items:
        tokens = preprocess_tokens(item["words"])
        tags = item["pos_tags"]
        preprocessed.append((tokens, tags))

    vocab = build_vocab(preprocessed)
    encoded = [vocab.encode(tokens, tags) for tokens, tags in preprocessed]
    return preprocessed, vocab, encoded

# Then in load_dataset_by_name:
loaders = {
    # ...
    "ontonotes": lambda: load_ontonotes(split=split, n=n),
}
```

## Troubleshooting

**Problem**: "Unknown dataset" error
**Solution**: Check spelling in config's `"dataset"` field matches the key in `load_dataset_by_name()`

**Problem**: Mismatched tokens and tags length
**Solution**: Add length check and filter:
```python
preprocessed = [(t, g) for t, g in preprocessed if len(t) == len(g)]
```

**Problem**: Memory error with large dataset
**Solution**: Use `n` parameter to load subset, or implement batched loading

**Problem**: Wrong tag format
**Solution**: Convert tags to strings: `tags = [str(tag) for tag in tags]`

## Need Help?

- Check existing loaders in `src/data/dataset.py` for reference
- See `docs/datasets.md` for dataset characteristics
- Run `python test_datasets.py` to verify all loaders work
