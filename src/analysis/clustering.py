"""Cluster embeddings and evaluate sense separation across layers."""

from typing import Mapping, List
from dataclasses import dataclass
from collections import defaultdict
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd  # pylint: disable=E0401
from sklearn.manifold import TSNE  # pylint: disable=E0401
import umap  # pylint: disable=E0401


@dataclass
class ClusterRow:
    """Result of clustering a single lemma."""

    sentence_id: int
    layer: int
    lemma: str
    label: str
    coords: np.ndarray


class ClusteringModel(ABC):
    """Base class for clustering models."""

    @abstractmethod
    def __init__(self, n, random_state):
        pass

    @abstractmethod
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings to 2D coordinates."""
        pass


class TSNEClusteringModel(ClusteringModel):
    """t-SNE clustering model."""

    def __init__(self, n, random_state=42):
        self.random_state = random_state
        self.n_components = 2
        self.perplexity = min(30, n - 1) if n > 1 else 1

        self.model = TSNE(
            n_components=self.n_components,
            random_state=self.random_state,
            perplexity=self.perplexity,
        )

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return self.model.fit_transform(embeddings)


class UMAPClusteringModel(ClusteringModel):
    """UMAP clustering model."""

    def __init__(self, n, random_state=42):
        self.random_state = random_state
        self.n_components = 2
        self.n_neighbors = min(15, n - 1) if n > 1 else 1

        self.model = umap.UMAP(
            n_components=self.n_components,
            random_state=self.random_state,
            n_neighbors=self.n_neighbors,
        )

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return self.model.fit_transform(embeddings)


def cluster_layer(
    lemma: str,
    layer: int,
    layer_df: pd.DataFrame,
    model_class: type[ClusteringModel],
    random_state: int = 42,
) -> Mapping[str, List[ClusterRow]]:
    """
    Return a dictionary of label -> list of ClusterRow after dimensionality reduction.
    Filters for rows corresponding to the given lemma.
    Pass in the ClusteringModel class to use.
    """
    # Filter layer_df for the given lemma
    layer_df = layer_df[layer_df["lemma"] == lemma]

    sentence_ids = layer_df["sentence_id"].values
    labels = layer_df["label"].values
    embeddings = np.stack(layer_df["embedding"].values)

    # Build model instance and fit
    model = model_class(len(embeddings), random_state=random_state)
    coords = model.transform(embeddings)

    # Build label -> ClusterRow mapping for this layer
    result = defaultdict(list)
    for sentence_id, label, point in zip(sentence_ids, labels, coords):
        result[label].append(
            ClusterRow(
                sentence_id=sentence_id,
                layer=layer,
                lemma=lemma,
                label=label,
                coords=point,
            )
        )

    return result


def cluster_all_layers(
    lemma: str,
    lemma_df: pd.DataFrame,
    model_class: type[ClusteringModel],
    random_state: int = 42,
) -> dict:
    """
    Return a dictionary of layer -> label -> list of ClusterRow after dimensionality reduction.
    Pass in the ClusteringModel class to use.
    """
    # Filter lemma_df for the given lemma
    lemma_df = lemma_df[lemma_df["lemma"] == lemma]

    # Cluster all layers
    result = defaultdict(lambda: defaultdict(list))
    for layer in sorted(lemma_df["layer"].unique()):
        layer_df = lemma_df[lemma_df["layer"] == layer]

        # Cluster this layer
        result[layer] = cluster(lemma, layer, layer_df, model_class, random_state)

    return result
