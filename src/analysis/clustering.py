"""Cluster embeddings and evaluate sense separation across layers."""

from typing import Mapping
from abc import ABC, abstractmethod

import numpy as np
from sklearn.manifold import TSNE
import umap

from src.types.clusters import (
    LemmaClusters,
    LayerClusters,
    LabelClusters,
    ClusterRecord,
)
from src.types.embeddings import LayerEmbeddings
from src.dataset import Database


class ClusteringModel(ABC):  # pylint: disable=R0903
    """Base class for clustering models."""

    @abstractmethod
    def __init__(self, n, random_state):
        pass

    @abstractmethod
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings to 2D coordinates."""
        raise NotImplementedError


class TSNEClusteringModel(ClusteringModel):  # pylint: disable=R0903
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


class UMAPClusteringModel(ClusteringModel):  # pylint: disable=R0903
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


def make_clusters(
    database: Database,
    model_class: type[ClusteringModel],
    random_state: int = 42,
) -> Mapping[str, LemmaClusters]:
    """
    Return a dictionary of lemma -> LemmaClusters after dimensionality reduction.
    """
    result: Mapping[str, LemmaClusters] = {}

    embeddings = database.embeddings_table.get_embeddings()

    for lemma, lemma_embeddings in embeddings.items():
        for layer, layer_embeddings in lemma_embeddings.layers.items():
            result.get(lemma, LemmaClusters(lemma=lemma)).layers[layer] = (
                _cluster_layer(
                    lemma, layer, layer_embeddings, model_class, random_state
                )
            )

    return result


def _cluster_layer(
    lemma: str,
    layer: int,
    layer_embeddings: LayerEmbeddings,
    model_class: type[ClusteringModel],
    random_state: int = 42,
) -> LayerClusters:
    """
    Return a dictionary of label -> list of ClusterRecord after dimensionality reduction.
    Filters for rows corresponding to the given lemma.
    Pass in the ClusteringModel class to use.
    """
    result: LayerClusters = LayerClusters(layer=layer)

    # Flatten the embeddings into a list of (sentence_id, label, embedding) tuples
    recs = [
        (rec.sentence_id, rec.label, rec.embedding)
        for label_emb in layer_embeddings.labels.values()
        for rec in label_emb.records
    ]
    sentence_ids, labels, embeddings = zip(*recs)

    # Build model instance and fit
    model = model_class(len(embeddings), random_state=random_state)
    coords = model.transform(embeddings)

    # Reorganize the clusters
    for sentence_id, label, coord in zip(sentence_ids, labels, coords):
        result.labels.get(label, LabelClusters(label=label)).records.append(
            ClusterRecord(
                sentence_id=sentence_id,
                layer=layer,
                lemma=lemma,
                label=label,
                coords=coord,
            )
        )

    return result


__all__ = [
    "make_clusters",
    "TSNEClusteringModel",
    "UMAPClusteringModel",
]
