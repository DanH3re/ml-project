# Datasets for POS Tagging

This document describes all available datasets for POS tagging experiments and provides recommendations for different use cases.

## Currently Implemented Datasets

### 1. Universal Dependencies - English EWT (ud)
- **Description**: The English Web Treebank (EWT) from Universal Dependencies. Contains text from blogs, newsgroups, emails, reviews, and Yahoo! Answers.
- **Size**: ~16,600 sentences
- **Tag Set**: Universal POS tags (17 tags: NOUN, VERB, ADJ, ADV, PRON, DET, ADP, NUM, CONJ, PRT, etc.)
- **Genre**: Web text (diverse, informal)
- **Best For**: Modern, informal text; cross-lingual comparisons
- **Usage**: `"dataset": "ud"`
- **Recommended maxlen**: 30-40

### 2. Brown Corpus (brown)
- **Description**: Classic balanced corpus from 1961 containing American English text across 15 genres
- **Size**: ~57,000 sentences
- **Tag Set**: Brown tag set (87 tags, mapped to universal tags in this implementation)
- **Genre**: News, fiction, government documents, academic prose, etc.
- **Best For**: Formal, edited text; baseline comparisons; large vocabulary studies
- **Usage**: `"dataset": "brown"`
- **Recommended maxlen**: 25-35
- **Note**: Use `"sentences": "max"` to load all sentences

### 3. CoNLL-2003 (conll2003)
- **Description**: News corpus from Reuters, primarily designed for NER but includes POS tags
- **Size**: ~23,500 sentences
- **Tag Set**: Penn Treebank tag set (45 tags)
- **Genre**: News articles
- **Best For**: News domain; joint NER+POS experiments
- **Usage**: `"dataset": "conll2003"`
- **Recommended maxlen**: 35-45
- **Note**: Longer sentences on average due to news article structure

### 4. Penn Treebank (ptb)
- **Description**: Wall Street Journal articles from the Penn Treebank corpus
- **Size**: ~3,900 sentences (via NLTK)
- **Tag Set**: Penn Treebank tag set (45 tags)
- **Genre**: Financial news (Wall Street Journal)
- **Best For**: Financial domain; fine-grained POS distinctions; academic comparisons
- **Usage**: `"dataset": "ptb"`
- **Recommended maxlen**: 30-40
- **Note**: Smaller dataset, good for quick experiments

### 5. GUM - Georgetown University Multilayer (gum)
- **Description**: Part of Universal Dependencies, contains diverse genres with rich annotation
- **Size**: ~10,000 sentences
- **Tag Set**: Universal POS tags (17 tags)
- **Genre**: News, interviews, how-tos, travel guides, fiction, biographies, academic writing
- **Best For**: Genre diversity; modern text across multiple domains
- **Usage**: `"dataset": "gum"`
- **Recommended maxlen**: 30-40

## Dataset Comparison Table

| Dataset | Size | Tags | Avg Length | Domain | Formality | Best Use Case |
|---------|------|------|------------|--------|-----------|---------------|
| UD EWT | 16.6k | 17 (UPOS) | ~25 | Web/General | Informal | Modern, diverse text |
| Brown | 57k | 17 (UPOS) | ~22 | Multi-genre | Formal | Large-scale, classic baseline |
| CoNLL-2003 | 23.5k | 45 (PTB) | ~30 | News | Formal | News domain, fine-grained tags |
| PTB | 3.9k | 45 (PTB) | ~25 | Finance | Formal | Financial news, quick tests |
| GUM | 10k | 17 (UPOS) | ~24 | Multi-genre | Mixed | Genre diversity |

## Usage Examples

### Basic Usage in Config Files

```json
{
  "name": "experiment_name",
  "dataset": "brown",
  "sentences": 2000,
  "maxlen": 30,
  ...
}
```

### Load All Sentences
```json
{
  "dataset": "brown",
  "sentences": "max",
  ...
}
```

### Dataset-Specific Recommendations

#### For Quick Prototyping
- **PTB** (small, fast to load)
```json
{"dataset": "ptb", "sentences": "max", "maxlen": 30}
```

#### For Production/Best Performance
- **Brown** (large, diverse, well-established)
```json
{"dataset": "brown", "sentences": "max", "maxlen": 30}
```

#### For Cross-Dataset Generalization Studies
Use matched tag sets:
- **UD EWT + GUM** (both use Universal POS tags)
```json
[
  {"dataset": "ud", "sentences": 2000},
  {"dataset": "gum", "sentences": 2000}
]
```

#### For Fine-Grained vs Coarse-Grained Comparison
- **CoNLL-2003** (45 PTB tags) vs **UD EWT** (17 UPOS tags)
```json
[
  {"dataset": "conll2003", "sentences": 2000},
  {"dataset": "ud", "sentences": 2000}
]
```

## Additional Dataset Suggestions

### Recommended for Future Implementation

1. **OntoNotes 5.0**
   - Size: ~75k sentences
   - Tags: Penn Treebank tag set
   - Domain: News, conversational telephone speech, weblogs, newsgroups, broadcast news, broadcast conversations
   - Why: Largest and most diverse English corpus; industry standard
   - How to add: `datasets` library: `load_dataset("ontonotes5", "english_v12")`

2. **Twitter POS (Tweebank)**
   - Size: ~3k tweets, ~57k tokens
   - Tags: Penn Treebank + Twitter-specific tags
   - Domain: Social media (Twitter)
   - Why: Test robustness on noisy, informal text with hashtags, emojis, @ mentions
   - How to add: `datasets` library: `load_dataset("universal_dependencies", "en_tweet")`

3. **WikiText POS**
   - Size: Variable (can be very large)
   - Domain: Wikipedia articles
   - Why: Clean, encyclopedic text; good for knowledge-rich domains
   - How to add: Custom processing needed from WikiText corpus

4. **Switchboard**
   - Size: ~200k utterances
   - Tags: Penn Treebank tag set
   - Domain: Conversational telephone speech
   - Why: Spoken language, very different from written text
   - How to add: Requires Switchboard corpus access via LDC

5. **ATIS (Air Travel Information System)**
   - Size: ~5k sentences
   - Tags: Penn Treebank tag set
   - Domain: Airline reservation dialogues
   - Why: Task-specific language, limited domain
   - How to add: Available through various NLP datasets

6. **Reddit POS**
   - Domain: Social media (Reddit)
   - Why: Modern informal text, memes, slang
   - How to add: Custom processing from Reddit data dumps

## Experimental Design Recommendations

### Hypothesis Testing

**H1: Model architecture comparison**
- Use same dataset across models: `"dataset": "ud"` for all configs

**H2: Domain robustness**
- Train on formal (CoNLL/PTB), test on informal (UD EWT)
- Requires custom data splits

**H3: Data efficiency**
- Same dataset, varying `"sentences"`: [100, 500, 1000, 2000, "max"]

**H4: Tag set granularity**
- Compare UPOS (17 tags) vs PTB (45 tags)
- Use: `{"dataset": "ud"}` vs `{"dataset": "conll2003"}`

**H5: Genre effects**
- Compare: Brown (mixed), PTB (finance), GUM (diverse), UD(web)

### Running Experiments

#### Single Dataset
```bash
python src/train_pos.py --config resources/configs/brown/baseline_comparison.json
```

#### All Datasets (using run_all_configs.py)
```bash
python src/run_all_configs.py \
  --configs-dir resources/configs \
  --use-gpu amd \
  --continue-on-error
```

This will run all configs in:
- `resources/configs/ud/*.json`
- `resources/configs/brown/*.json`
- `resources/configs/conll2003/*.json`
- `resources/configs/ptb/*.json`
- `resources/configs/gum/*.json`

## Tag Set Details

### Universal POS Tags (17 tags)
Used by: UD EWT, Brown (converted), GUM

```
ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X
```

### Penn Treebank Tags (45 tags)
Used by: CoNLL-2003, PTB

```
CC, CD, DT, EX, FW, IN, JJ, JJR, JJS, LS, MD, NN, NNS, NNP, NNPS, PDT, POS, PRP, PRP$, RB, RBR, RBS, RP, SYM, TO, UH, VB, VBD, VBG, VBN, VBP, VBZ, WDT, WP, WP$, WRB, #, $, '', (, ), ,, ., :, ``
```

## Notes

- All loaders normalize tokens (lowercase, strip whitespace)
- Special tokens: `<PAD>` (id=0) and `<UNK>` for unknown words
- Padding masks are automatically generated and used during evaluation
- Dataset caching is implemented – multiple configs using the same dataset won't reload it

## References

- Universal Dependencies: https://universaldependencies.org/
- Brown Corpus: Francis & Kucera (1979)
- CoNLL-2003: Tjong Kim Sang & De Meulder (2003)
- Penn Treebank: Marcus et al. (1993)
- GUM Corpus: Zeldes (2017)
