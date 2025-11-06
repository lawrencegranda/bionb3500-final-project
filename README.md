# BIONB 3500 Final Project

Understanding Contextual Meaning in Transformer Embeddings

## Overview

This project investigates how BERT embeddings reflect context-dependent meaning and whether transformer models implicitly separate different senses of the same word.

## Setup

```bash
poetry install
```

## Workflow

The complete analysis pipeline consists of 5 steps:

```bash
# 1. Build sense map (maps words to their different senses)
python scripts/build_sense_map.py

# 2. Extract sentences from corpora and save to database
python scripts/save_corpora.py

# 3. View dataset statistics
python scripts/summarise.py -d config/data.yaml

# 4. Extract BERT embeddings for all layers
python scripts/extract_embeddings.py -d config/data.yaml --layer -1

# 5. Analyze clustering and sense separation
python scripts/cluster_analysis.py -d config/data.yaml --visualize
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

### 5. Cluster Analysis

Evaluate sense separation and clustering quality across layers:

```bash
python scripts/cluster_analysis.py -d config/data.yaml --visualize
```

Options:

- `-d, --data-config-path`: Path to YAML configuration file (required)
- `--lemma`: Analyze specific lemma only (default: all lemmas)
- `--method`: Clustering method - `kmeans`, `hdbscan`, or `both` (default: `both`)
- `--visualize`: Generate t-SNE and UMAP visualizations
- `--output-dir`: Directory to save results (default: `results/metrics`)

**Output**:

- `clustering_metrics.csv`: Detailed metrics for each lemma/layer/method
- `{lemma}_metrics.png`: Line plots showing how metrics vary across layers
- `{lemma}_tsne.png`: t-SNE visualizations for selected layers
- `{lemma}_umap.png`: UMAP visualizations for selected layers

**Metrics Evaluated**:

- **Silhouette Score** (higher = better clustering)
- **Adjusted Rand Index (ARI)** (measures agreement with true labels)
- **Normalized Mutual Information (NMI)** (information shared with true labels)
- **Davies-Bouldin Index** (lower = better separation)
- **Calinski-Harabasz Score** (higher = better defined clusters)

This directly tests the hypotheses from the proposal:

- **H1**: Whether embeddings naturally cluster by sense without supervision
- **H2**: Whether sense differentiation peaks in middle-to-upper layers

## Interpreting Results

### Clustering Metrics

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
