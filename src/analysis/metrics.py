"""Evaluate sense separation metrics across layers."""

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

import warnings
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from src.types.metrics import (
    LemmaMetrics,
    LayerMetrics,
    MetricRecord,
)
from src.types.embeddings import LayerEmbeddings
from src.dataset import Database

# Suppress warnings in this file
warnings.filterwarnings("ignore", category=UserWarning)


class MetricModel(ABC):  # pylint: disable=R0903
    """Base class for clustering evaluation metrics."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def compute(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Compute metric score for embeddings and labels."""
        raise NotImplementedError


class SilhouetteMetric(MetricModel):  # pylint: disable=R0903
    """Silhouette coefficient metric."""

    def __init__(self):
        self.name = "silhouette"

    def compute(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Compute silhouette score."""
        if len(set(labels)) < 2:
            return 0.0
        return float(silhouette_score(embeddings, labels))


class AdjustedRandMetric(MetricModel):  # pylint: disable=R0903
    """Adjusted Rand Index metric."""

    def __init__(self):
        self.name = "adjusted_rand"

    def compute(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Compute adjusted rand index (requires true labels)."""
        # This metric requires both predicted and true labels
        # For now, return 0.0 as placeholder
        return 0.0


class NormalizedMutualInfoMetric(MetricModel):  # pylint: disable=R0903
    """Normalized Mutual Information metric."""

    def __init__(self):
        self.name = "normalized_mutual_info"

    def compute(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Compute normalized mutual information (requires true labels)."""
        # This metric requires both predicted and true labels
        # For now, return 0.0 as placeholder
        return 0.0


class DaviesBouldinMetric(MetricModel):  # pylint: disable=R0903
    """Davies-Bouldin Index metric."""

    def __init__(self):
        self.name = "davies_bouldin"

    def compute(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Compute Davies-Bouldin index."""
        if len(set(labels)) < 2:
            return 0.0
        return float(davies_bouldin_score(embeddings, labels))


class CalinskiHarabaszMetric(MetricModel):  # pylint: disable=R0903
    """Calinski-Harabasz Index metric."""

    def __init__(self):
        self.name = "calinski_harabasz"

    def compute(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Compute Calinski-Harabasz index."""
        if len(set(labels)) < 2:
            return 0.0
        return float(calinski_harabasz_score(embeddings, labels))


def make_metrics(
    database: Database,
    metric_classes: Sequence[type[MetricModel]],
) -> Mapping[str, LemmaMetrics]:
    """
    Return a dictionary of lemma -> LemmaMetrics after computing metrics.
    """
    result: Mapping[str, LemmaMetrics] = {}

    embeddings = database.embeddings_table.get_embeddings()

    for lemma, lemma_embeddings in embeddings.items():
        if lemma not in result:
            result[lemma] = LemmaMetrics(lemma=lemma)

        for layer, layer_embeddings in lemma_embeddings.layers.items():
            result[lemma].layers[layer] = _compute_layer_metrics(
                lemma, layer, layer_embeddings, metric_classes
            )

    return result


def _compute_layer_metrics(
    lemma: str,
    layer: int,
    layer_embeddings: LayerEmbeddings,
    metric_classes: Sequence[type[MetricModel]],
) -> LayerMetrics:
    """
    Compute all metrics for a single layer.
    Returns a LayerMetrics object with metric name -> MetricRecord mapping.
    """
    result: LayerMetrics = LayerMetrics(layer=layer)

    # Flatten the embeddings into arrays
    recs = [
        (label_emb.label, rec.embedding)
        for label_emb in layer_embeddings.labels.values()
        for rec in label_emb.records
    ]

    if len(recs) < 2:
        # Not enough data to compute metrics
        return result

    labels = np.array([rec[0] for rec in recs])
    embeddings = np.array([rec[1] for rec in recs])

    # Compute each metric
    for metric_class in metric_classes:
        metric = metric_class()
        value = metric.compute(embeddings, labels)

        metric_record = MetricRecord(
            lemma=lemma,
            layer=layer,
            metric=metric.name,
            value=value,
        )

        result.labels[metric.name] = metric_record

    return result


metric_models: Mapping[str, type[MetricModel]] = {
    "silhouette": SilhouetteMetric,
    "adjusted_rand": AdjustedRandMetric,
    "normalized_mutual_info": NormalizedMutualInfoMetric,
    "davies_bouldin": DaviesBouldinMetric,
    "calinski_harabasz": CalinskiHarabaszMetric,
}


__all__ = [
    "metric_models",
    "make_metrics",
]
