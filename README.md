# BIONB 3500 Final Project

Understanding Contextual Meaning in Transformer Embeddings

## Overview

This project investigates how BERT embeddings reflect context-dependent meaning and whether transformer models implicitly separate different senses of the same word.

## Setup

```bash
poetry install
```

## Usage

### 1. Build Sense Map

```bash
python scripts/build_sense_map.py
```

### 2. Save Corpora to Database

```bash
python scripts/save_corpora.py
```

### 3. Summarize Dataset Statistics

View dataset and embedding statistics:

```bash
python scripts/summarise.py -d config/data.yaml
```

This will display:

- Dataset statistics by label, synset, and source
- Embedding counts per layer
- Mean embedding analysis per label and layer (with mean norm and std deviation)
- Overall embedding statistics across all labels

### 4. Extract BERT Embeddings

Extract embeddings for all sentences in the dataset:

```bash
python scripts/extract_embeddings.py -d config/data.yaml
```

Options:

- `-d, --data-config-path`: Path to YAML configuration file (required)
- `--model`: HuggingFace model identifier (default: `bert-base-uncased`)
- `--layer`: BERT layer to extract up to, -1 for all layers (default: `-1`)

**Note**: The script extracts embeddings from **all layers up to the specified layer**. For example:

- `--layer -1`: Extracts all 13 layers (0-12 for BERT-base)
- `--layer 6`: Extracts layers 0 through 6
- `--layer 0`: Extracts only layer 0

## Database Schema

### Sentences Table

- `id`: Unique sentence identifier
- `lemma`: Target lemma
- `text`: Sentence text
- `label`: Sense label
- `synset`: WordNet synset
- `source`: Source corpus

### Embeddings Table

- `sentence_id`: Foreign key to sentences.id
- `layer`: Layer number extracted from
- `embedding`: BERT embedding vector (BLOB)
- **Primary Key**: (`sentence_id`, `layer`) - Each sentence can have multiple embeddings, one per layer

## Project Structure

```
.
├── config/          # Configuration files
├── data/            # Datasets and raw corpora
├── latex/           # Project documentation
├── results/         # Output files (embeddings, plots, metrics)
├── scripts/         # Executable scripts
└── src/             # Source code
    ├── dataset/     # Dataset loading and management
    ├── utils/       # Utility functions
    └── embeddings.py # BERT embedding extraction
```
