# BIONB 3500 Final Project

Understanding Contextual Meaning in Transformer Embeddings

## Overview

This project investigates how BERT embeddings reflect context-dependent meaning and whether transformer models implicitly separate different senses of the same word.

## Setup

```bash
poetry install
```

## Workflow

### 1. Configure Your Analysis

Edit configuration files to specify target words and parameters:

- `config/words.yaml` - Define target words and their sense definitions
- `config/data.yaml` - Set paths, corpora, models, and clustering layers

### 2. Build Sense Map

Generate WordNet sense mappings from gloss substrings:

```bash
python -m scripts.build_sense_map -d config/data.yaml
```

This creates `results/senses/sense_map.json` mapping words → sense labels → WordNet keys.

### 3. Save Corpora to Database

Filter and persist SemCor sentences to SQLite database:

```bash
python -m scripts.save_corpora -d config/data.yaml
```

This creates `data/dataset.db` with filtered sentences for your target words.

### 4. Run Complete Analysis Pipeline

Execute the full analysis workflow (recommended):

```bash
python -m scripts.run_analysis -d config/data.yaml
```

This automatically:

- Summarizes the dataset statistics
- Extracts embeddings for all configured models
- Generates clustering visualizations

**OR** run individual steps:

```bash
# Summarize dataset statistics
python -m scripts.summarise -d config/data.yaml

# Extract embeddings for a specific model
python -m scripts.extract_embeddings -d config/data.yaml --model bert-base-uncased

# Generate clustering plots for a specific model
python -m scripts.plot_clusters -d config/data.yaml --model bert-base-uncased
```

### 5. View Results

- **Plots**: `results/plots/{model_name}/` - UMAP/t-SNE visualizations
- **Metrics**: `results/metrics/` - Clustering quality metrics (if computed)
- **Database**: `data/dataset.db` - Raw sentences and embeddings

## Interpreting Results

### Metrics

1. **Silhouette Score** (range: -1 to 1)

   - Measures how similar embeddings are to their own cluster vs. other clusters
   - Higher values indicate better-defined clusters
   - Values > 0.5 suggest strong clustering

2. **Adjusted Rand Index (ARI)** (range: -1 to 1)

   - Measures agreement between discovered clusters and true sense labels
   - 1.0 = perfect agreement, 0.0 = random clustering
   - Tests **H1**: Do clusters match semantic senses?

3. **Normalized Mutual Information (NMI)** (range: 0 to 1)

   - Measures information shared between clusters and true labels
   - Higher values = clusters capture sense distinctions
   - Complements ARI for hypothesis testing

4. **Davies-Bouldin Index** (range: 0 to ∞)

   - Lower is better (well-separated clusters)
   - Values < 1.0 indicate good separation

5. **Calinski-Harabasz Score** (range: 0 to ∞)
   - Higher is better (dense, well-separated clusters)
   - Useful for comparing relative quality across layers

### Expected Patterns (from H2)

According to the **Layer-wise Semantic Specialization** hypothesis:

- **Lower layers (0-3)**: Lower scores (surface/syntactic features)
- **Middle layers (4-8)**: Peak scores (semantic features emerge)
- **Upper layers (9-12)**: Scores may taper (task-specific tuning)

Look for peaks in ARI, NMI, and Silhouette scores in middle-to-upper layers to confirm H2.

## Project Structure

```
.
├── config/
│   ├── data.yaml           # Main configuration (paths, models, parameters)
│   └── words.yaml          # Target words and sense definitions
├── data/
│   ├── raw/                # Raw XML corpora (SemCor, WNGT, etc.)
│   └── dataset.db          # Generated SQLite database
├── latex/                  # LaTeX documentation and figures
├── results/
│   ├── metrics/            # Computed clustering metrics
│   ├── plots/              # Visualization outputs (by model)
│   │   ├── bert-base-uncased/
│   │   └── distilbert-base-uncased/
│   └── senses/
│       └── sense_map.json  # Generated sense mappings
├── scripts/
│   ├── helpers.py          # Shared configuration utilities
│   ├── build_sense_map.py  # Step 1: Build WordNet sense map
│   ├── save_corpora.py     # Step 2: Filter and save sentences
│   ├── summarise.py        # Summarize dataset statistics
│   ├── extract_embeddings.py # Extract BERT embeddings
│   ├── plot_clusters.py    # Generate visualizations
│   └── run_analysis.py     # Run complete pipeline
└── src/
    ├── analysis/
    │   ├── clustering.py   # Dimensionality reduction (UMAP, t-SNE)
    │   └── metrics.py      # Clustering evaluation metrics
    ├── builders/
    │   ├── corpora.py      # Corpus loading and filtering
    │   └── sense_map.py    # Sense map construction
    ├── dataset/
    │   ├── database.py     # SQLite database interface
    │   ├── embeddings.py   # BERT embedding extraction
    │   └── sentences.py    # Sentence storage and retrieval
    └── types/              # Type definitions and data structures
        ├── clusters.py
        ├── embeddings.py
        ├── metrics.py
        ├── sentences.py
        └── senses.py
```
